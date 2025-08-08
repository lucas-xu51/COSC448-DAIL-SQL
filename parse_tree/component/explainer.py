from typing import List, Dict, Optional, Tuple
from .query import Query, SchemaGraph
from .stanzaParser import ParseTree, ParseTreeNode
from .tree_structure_adjustor import Tree, TreeNode
import re

class Explainer:
    @staticmethod
    def explain(query: Query) -> None:
        """Generate natural language explanations for each adjusted tree of the query"""
        if not hasattr(query, 'adjusted_trees') or not query.adjusted_trees:
            print("No adjusted trees available for explanation")
            return
        
        query.nl_sentences = []  # Store generated natural language sentences
        for i, tree in enumerate(query.adjusted_trees):
            try:
                nl = Explainer.explain_tree(tree)
                if nl:
                    query.nl_sentences.append(nl)
            except Exception as e:
                print(f"Error generating explanation: {e}")
    
    @staticmethod
    def explain_tree(tree: ParseTree) -> Optional[Dict]:
        """Convert the parse tree into a natural language sentence, handling different tree structures more generally"""
        nl = {
            'words': [],       # Generated natural language word list
            'nodes': [],       # Corresponding parse tree nodes
            'is_implicit': []  # Marks whether each word was implicitly added
        }
        
        # Try to find the root node (supports "ROOT" label or the actual root node)
        root = None
        if tree.nodes and tree.nodes[0].label == "ROOT":
            root = tree.nodes[0]
        else:
            # Try to find the actual root node (node whose relation is 'root')
            for node in tree.nodes:
                if hasattr(node, 'rel') and node.rel == 'root':
                    root = node
                    break
        
        if not root:
            print("Warning: Root node not found")
            return None
        
        # Handle the root node (if it's of type ROOT, use its first child as the operation)
        if root.label == "ROOT":
            if not root.children:
                return None
            root_op = root.children[0]
            Explainer._add_node(nl, root_op, root_op.label, False)
        else:
            # Directly use the identified root node
            Explainer._add_node(nl, root, root.label, False)
        
        # Find the core noun phrase (NT type)
        core_nt = None
        if root.label == "ROOT" and root.children:
            # Search for NT in the children of ROOT
            for child in root.children:
                if child.token_type == "NT":
                    core_nt = child
                    break
        else:
            # Search among the children of the actual root node
            for child in root.children:
                if child.token_type == "NT":
                    core_nt = child
                    break
        
        if not core_nt:
            # Try a more relaxed search
            for node in tree.nodes:
                if node.token_type == "NT":
                    core_nt = node
                    break
        
        if not core_nt:
            print("Warning: Core noun phrase not found")
            return None
        
        # Process the core noun phrase
        add_the = True  # Add "the" by default
        Explainer._add_core_nt(core_nt, add_the, nl)
        
        # Process condition clauses (WHERE part)
        has_where = False
        for node in tree.nodes:
            # Find operator nodes (OT type)
            if node.token_type != "OT" or len(getattr(node, 'children', [])) != 2:
                continue
            
            # Add "where" keyword
            if not has_where:
                Explainer._add_node(nl, None, "where", False)
                has_where = True
            
            left, right = node.children
            
            # Process left child node
            if left.token_type == "FT":
                Explainer._add_node(nl, left, left.label, False)
            
            # Process core noun phrase
            if left.token_type == "NT":
                Explainer._add_core_nt(left, True, nl)
            
            # Add operator
            if node.function != "=":
                ot_text = Explainer._to_ot(node)
                Explainer._add_node(nl, node, f"is {ot_text}", False)
            else:
                Explainer._add_node(nl, node, "is", False)
            
            # Process right child node
            if right.token_type == "NT":
                Explainer._add_core_nt(right, True, nl)
            elif right.token_type == "VTTEXT":
                Explainer._add_node(nl, right, right.label, False)
        
        return nl
    
    @staticmethod
    def _add_core_nt(core_nt: ParseTreeNode, add_the: bool, nl: Dict) -> None:
        """Recursively add the core noun phrase and its child nodes, handling different structures more generally"""
        node_stack = [core_nt]
        
        while node_stack:
            current = node_stack.pop()
            label = ""
            
            # Add "the" prefix
            if current == core_nt and add_the:
                label += "the "
            
            # Handle prepositions
            if hasattr(current, 'prep') and current.prep:
                label += f"{current.prep} "
            
            # Add quantifier
            if hasattr(current, 'QT') and current.QT:
                label += f"{current.QT} "
            
            # Add node label
            if len(current.label.split()) > 1:
                label += f"\"{current.label}\""
            else:
                label += current.label
            
            # Add to natural language result
            is_added = hasattr(current, 'is_added') and current.is_added
            Explainer._add_node(nl, current, label, is_added)
            
            # Push child nodes onto the stack (in reverse order to maintain correct order)
            for child in reversed(getattr(current, 'children', [])):
                node_stack.append(child)
    
    @staticmethod
    def _add_node(nl: Dict, node: Optional[ParseTreeNode], word: str, is_implicit: bool) -> None:
        """Add a node and its corresponding natural language word"""
        nl['words'].append(word)
        nl['nodes'].append(node)
        nl['is_implicit'].append(is_implicit)
    
    @staticmethod
    def _sub_tree_contains(node: ParseTreeNode, target: ParseTreeNode) -> bool:
        """Check whether the subtree contains the target node"""
        if node == target:
            return True
        for child in getattr(node, 'children', []):
            if Explainer._sub_tree_contains(child, target):
                return True
        return False
    
    @staticmethod
    def _lemmatize(word: str) -> str:
        """Simple lemmatization"""
        if word.endswith('s'):
            return word[:-1]
        return word
    
    @staticmethod
    def _to_ot(node: ParseTreeNode) -> str:
        """Convert operator node to natural language representation"""
        function_map = {
            ">": "greater than",
            "<": "less than",
            ">=": "greater than or equal to",
            "<=": "less than or equal to",
            "!=": "not equal to",
            "LIKE": "like",
            "IN": "in"
        }
        return function_map.get(getattr(node, 'function', ''), getattr(node, 'label', ''))