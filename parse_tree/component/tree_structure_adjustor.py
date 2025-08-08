import copy
from typing import List, Dict, Optional, Tuple
from .query import Query, SchemaGraph  # Ensure SchemaGraph is imported
from .stanzaParser import ParseTree, ParseTreeNode

class Tree:
    """Temporary tree class for tree structure adjustments (extends ParseTree functionality)"""
    def __init__(self, parse_tree: ParseTree):
        self.all_nodes = [self._convert_node(n) for n in parse_tree.nodes]  # Convert to TreeNode
        self.root = self.all_nodes[0] if self.all_nodes else None
        self.cost = 0  # Tree structure cost (number of adjustments)
        self.invalid = 0  # Invalidity score (0 means valid)
        self.weight = 0.0  # Tree weight (effectiveness score)
        self.hash_num = 0  # Tree hash value (for deduplication)
        self._init_parents_and_children()

    def _convert_node(self, parse_node: ParseTreeNode) -> 'TreeNode':
            node = TreeNode(
                node_id=parse_node.wordOrder,
                label=parse_node.label,
                token_type=parse_node.token_type,
                function=getattr(parse_node, 'function', ""),
                parent=None
            )
            node.mapped_elements = getattr(parse_node, 'mapped_elements', [])
            # Save original node's children wordOrder (key: for establishing parent-child relationships)
            node.original_children = [child.wordOrder for child in parse_node.children]
            return node

    def _init_parents_and_children(self):
        node_map = {n.node_id: n for n in self.all_nodes}
        for node in self.all_nodes:
            # Add children to current node (based on original parse tree's children info)
            for child_id in node.original_children:
                child_node = node_map.get(child_id)
                if child_node:
                    node.children.append(child_node)
                    child_node.parent = node  # Also set child's parent node
        # Root node is the one without a parent
        self.root = next((n for n in self.all_nodes if n.parent is None), None)

    def move_sub_tree(self, node: 'TreeNode', new_parent: 'TreeNode') -> bool:
        """Move node to a new parent node"""
        if node == self.root:
            return False  # Cannot move root node
        # Remove from original parent
        if node.parent:
            node.parent.children.remove(node)
        # Add to new parent
        node.parent = new_parent
        new_parent.children.append(node)
        return True

    def add_equal(self):
        """Add equals operator to nodes that need it (special handling for max/min functions)"""
        for node in self.all_nodes:
            if node.function in ["max", "min"] and not any(c.token_type == "OT" for c in node.children):
                # Add an equals operator as child node
                equal_node = TreeNode(
                    node_id=max(n.node_id for n in self.all_nodes) + 1,
                    label="=",
                    token_type="OT",
                    function="=",
                    parent=node
                )
                self.all_nodes.append(equal_node)
                node.children.append(equal_node)

    def tree_evaluation(self, schema_graph: SchemaGraph, query: Query):
        """Evaluate tree structure validity (based on database schema)"""
        self.invalid = 0
        self.weight = 0.0
        # 1. Check if function nodes have required children
        for node in self.all_nodes:
            if node.function in ["avg", "sum", "max", "min"]:
                if not node.children:
                    self.invalid += 1  # Function nodes must have children
                else:
                    self.weight += 0.5  # Bonus for valid function nodes
            # 2. Check if operator nodes have exactly two children
            if node.token_type == "OT" and len(node.children) != 2:
                self.invalid += 1
            # 3. Check if node mappings match database relationships
            if hasattr(node, 'mapped_elements') and node.mapped_elements:
                elem = node.mapped_elements[0].schema_element
                # Check if current node's mapping matches parent's mapping with database relationships
                if node.parent and hasattr(node.parent, 'mapped_elements') and node.parent.mapped_elements:
                    parent_elem = node.parent.mapped_elements[0].schema_element
                    # Get all elements related to parent element from schema_graph
                    related_elements = schema_graph.get_related_elements(parent_elem)
                    if elem in related_elements:
                        self.weight += 0.3  # Bonus for matching relationships

    def hash_tree_to_number(self):
        """Calculate tree hash value (for deduplication)"""
        node_hashes = [f"{n.node_id}:{n.label}:{n.parent.node_id if n.parent else 'None'}" for n in self.all_nodes]
        self.hash_num = hash("|".join(sorted(node_hashes)))

    def __lt__(self, other: 'Tree') -> bool:
        """Sorting: lower invalidity first, lower cost first"""
        if self.invalid != other.invalid:
            return self.invalid < other.invalid
        return (self.weight * 10000 - self.cost) > (other.weight * 10000 - other.cost)


class TreeNode:
    """Node class for tree structure adjustments (extends ParseTreeNode functionality)"""
    def __init__(self, node_id: int, label: str, token_type: str, function: str, parent: Optional['TreeNode']):
        self.node_id = node_id  # Node ID
        self.label = label  # Node text
        self.token_type = token_type  # Node type (NT/VTTEXT etc)
        self.function = function  # Function/operator (avg/=/etc)
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.mapped_elements = []  # Mapped database elements

    def __repr__(self) -> str:
        return f"TreeNode({self.label}, type={self.token_type}, id={self.node_id})"


class TreeStructureAdjustor:
    @staticmethod
    def tree_structure_adjust(query: Query, schema_graph: SchemaGraph) -> None:
        """Main function for tree structure adjustment (receives SchemaGraph instead of db)"""
        print("\n----- Step 8: Tree Structure Adjustment -----")
        if not query.parse_tree or not query.parse_tree.nodes:
            print("Parse tree is empty, cannot adjust structure")
            return
        
        # Initialize adjusted trees list
        query.adjusting_trees = []
        pre_trees = {}  # Hash table for deduplication (hash value -> cost)

        # Execute adjustment process (pass schema_graph)
        TreeStructureAdjustor.adjust(query, schema_graph, add_equal=False, pre_trees=pre_trees)
        
        # Check if supplemental adjustment is needed (for max/min functions)
        has_max_min = any(node.function in ["max", "min"] for node in query.parse_tree.nodes if hasattr(node, 'function'))
        if (not query.adjusting_trees or (query.adjusting_trees and query.adjusting_trees[0].cost > 3)) and has_max_min:
            TreeStructureAdjustor.adjust(query, schema_graph, add_equal=True, pre_trees=pre_trees)

        # Sort and deduplicate adjusted trees
        adjusting_trees = query.adjusting_trees
        adjusting_trees.sort()
        TreeStructureAdjustor._deduplicate_trees(adjusting_trees)

        # Build final adjusted trees (convert to ParseTree format)
        TreeStructureAdjustor.build_adjusted_trees(query)
        print(f"Tree structure adjustment complete, kept {len(query.adjusted_trees)} valid tree structures")

    @staticmethod
    def pre_adjust(tree: Tree) -> None:
        """Pre-adjustment: clean up obviously incorrect node structures"""
        # Handle function nodes (avg/sum) without parameters
        for node in tree.all_nodes:
            if node.function in ["avg", "sum"] and not node.children:
                if node.parent:
                    tree.move_sub_tree(node, node.parent)
                    print(f"Pre-adjust: moved function node {node.label} without parameters under parent")
            # Handle operator nodes (OT) without parameters
            if node.token_type == "OT" and not node.children and node.parent:
                tree.move_sub_tree(node, node.parent)
                print(f"Pre-adjust: moved operator node {node.label} without parameters under parent")

    @staticmethod
    def adjust(query: Query, schema_graph: SchemaGraph, add_equal: bool, pre_trees: Dict[int, int]) -> None:
        """Expand and evaluate tree structures (using schema_graph)"""
        # Create initial tree from original parse tree
        initial_tree = Tree(query.parse_tree)
        TreeStructureAdjustor.pre_adjust(initial_tree)
        
        # Handle special cases for max/min functions (add equals operator)
        if add_equal:
            initial_tree.add_equal()
        
        # Evaluate initial tree validity (pass schema_graph)
        initial_tree.tree_evaluation(schema_graph, query)
        initial_tree.hash_tree_to_number()
        
        # Save valid initial tree
        if initial_tree.invalid == 0:
            query.adjusting_trees.append(initial_tree)
        
        # Expand tree structures using queue (breadth-first search)
        queue = [initial_tree]
        pre_trees[initial_tree.hash_num] = initial_tree.cost
        
        # Limit queue size (avoid excessive expansion)
        while queue and len(queue) < 100:
            current_tree = queue.pop(0)
            
            # Expand current tree, generate new possible structures (pass schema_graph)
            extended_trees = TreeStructureAdjustor.extend(current_tree, schema_graph, query)
            for new_tree in extended_trees:
                new_tree.hash_tree_to_number()
                # Deduplicate or update better trees (lower cost)
                if new_tree.hash_num in pre_trees:
                    if pre_trees[new_tree.hash_num] > new_tree.cost:
                        pre_trees[new_tree.hash_num] = new_tree.cost
                else:
                    queue.append(new_tree)
                    pre_trees[new_tree.hash_num] = new_tree.cost
                    if new_tree.invalid == 0:
                        query.adjusting_trees.append(new_tree)

    @staticmethod
    def extend(current_tree: Tree, schema_graph: SchemaGraph, query: Query) -> List[Tree]:
        """Expand tree structure: generate new trees by moving nodes"""
        extended_trees = []
        if current_tree.cost > 4:  # Limit maximum cost (number of adjustments)
            return extended_trees
        
        # Try moving each node under other nodes to generate new trees
        for node in current_tree.all_nodes[1:]:  # Skip root node
            extended = TreeStructureAdjustor.extend_node(current_tree, node, schema_graph, query)
            extended_trees.extend(extended)
        return extended_trees

    @staticmethod
    def extend_node(current_tree: Tree, node: TreeNode, schema_graph: SchemaGraph, query: Query) -> List[Tree]:
        """Move single node to generate new trees (using schema_graph for evaluation)"""
        extended_trees = []
        # Try moving node under each other node
        for target_parent in current_tree.all_nodes:
            if target_parent.node_id == node.node_id:
                continue  # Cannot move under itself
            # Clone current tree
            new_tree = copy.deepcopy(current_tree)
            new_tree.cost += 1  # Increment cost
            # Find corresponding nodes in new tree
            new_node = next(n for n in new_tree.all_nodes if n.node_id == node.node_id)
            new_target_parent = next(n for n in new_tree.all_nodes if n.node_id == target_parent.node_id)
            # Move node
            if new_tree.move_sub_tree(new_node, new_target_parent):
                # Evaluate new tree (pass schema_graph)
                new_tree.tree_evaluation(schema_graph, query)
                # Keep better trees
                if (new_tree.invalid < current_tree.invalid or 
                    (new_tree.invalid == current_tree.invalid and 
                     new_tree.weight * 10000 - new_tree.cost > current_tree.weight * 10000 - current_tree.cost)):
                    extended_trees.append(new_tree)
        return extended_trees

    @staticmethod
    def _deduplicate_trees(trees: List[Tree]) -> None:
        """Remove duplicate tree structures (via hash values)"""
        seen_hashes = set()
        i = 0
        while i < len(trees):
            tree = trees[i]
            if tree.hash_num in seen_hashes:
                trees.pop(i)
            else:
                seen_hashes.add(tree.hash_num)
                i += 1

    @staticmethod
    def build_adjusted_trees(query: Query) -> None:
        """Convert adjusted trees to ParseTree format"""
        query.adjusted_trees = []
        if not hasattr(query, 'adjusting_trees'):
            return  # Return directly if no adjusted trees
        
        adjusting_trees = query.adjusting_trees[:5]  # Keep top 5 best trees
        
        for adj_tree in adjusting_trees:
            # Clone original parse tree as base
            adjusted_tree = copy.deepcopy(query.parse_tree)
            # Add missing nodes
            node_id_map = {n.wordOrder: n for n in adjusted_tree.nodes}
            for tn in adj_tree.all_nodes:
                if tn.node_id not in node_id_map:
                    # Create new ParseTreeNode to add
                    new_node = ParseTreeNode(
                        label=tn.label,
                        word_order=tn.node_id,
                        pos="",
                        relationship="",
                        parent=None
                    )
                    new_node.token_type = tn.token_type
                    new_node.function = tn.function
                    new_node.mapped_elements = tn.mapped_elements
                    adjusted_tree.nodes.append(new_node)
                    node_id_map[tn.node_id] = new_node
            
            # Rebuild parent-child relationships
            for tn in adj_tree.all_nodes:
                parse_node = node_id_map[tn.node_id]
                # Update parent node
                if tn.parent:
                    parse_node.parent = node_id_map.get(tn.parent.node_id)
                # Clear and rebuild children
                parse_node.children = []
                for child_tn in tn.children:
                    child_node = node_id_map.get(child_tn.node_id)
                    if child_node:
                        parse_node.children.append(child_node)
            
            # Fix root node
            adjusted_tree.root = next((n for n in adjusted_tree.nodes if n.parent is None), None)
            query.adjusted_trees.append(adjusted_tree)
        
        # Set default tree to use
        if query.adjusted_trees:
            query.query_tree = query.adjusted_trees[0]
            print("Selected best adjusted tree as query tree")