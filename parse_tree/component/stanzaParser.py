import stanza
from typing import List, Dict, Optional, Tuple
from .query import Query  # Import Query class from query.py

class ParseTreeNode:
    """Syntax parse tree node"""
    def __init__(self, label: str, word_order: int, pos: str, 
                 relationship: str, parent: Optional['ParseTreeNode'] = None):
        self.label = label           # Node text content
        self.wordOrder = word_order  # Word position in sentence (1-based)
        self.pos = pos               # Part-of-speech tag
        self.relationship = relationship  # Dependency relation with parent node
        self.parent = parent         # Reference to parent node
        self.children = []           # List of child nodes
        self.leftRel = None          # Coordination relation marker

    def __repr__(self) -> str:
        return f"Node({self.label}, pos={self.pos}, rel={self.relationship})"

class ParseTree:
    """Syntax parse tree"""
    def __init__(self):
        self.root = None  # Root node
        self.nodes = []   # List of all nodes

    def build_node(self, node_info: Tuple[str, str, str, str, str]) -> bool:
        """Build and add node to tree based on tree table entry"""
        dep_index, dep_value, pos, gov_index, relationship = node_info
        dep_index = int(dep_index)
        gov_index = int(gov_index)
        
        # Create new node
        new_node = ParseTreeNode(
            label=dep_value,
            word_order=dep_index,
            pos=pos,
            relationship=relationship
        )
        
        # Handle root node
        if gov_index == 0:
            self.root = new_node
            self.nodes.append(new_node)
            return True
        
        # Find parent node
        parent_node = self.search_node_by_order(gov_index)
        if parent_node:
            new_node.parent = parent_node
            parent_node.children.append(new_node)
            self.nodes.append(new_node)
            return True
        
        return False  # Parent node not found, try again later

    def search_node_by_order(self, word_order: int) -> Optional[ParseTreeNode]:
        """Search node by word order"""
        for node in self.nodes:
            if node.wordOrder == word_order:
                return node
        return None

    def print_tree(self, node: Optional[ParseTreeNode] = None, level: int = 0):
        """Recursively print tree structure (for debugging)"""
        if node is None:
            node = self.root
        
        indent = "  " * level
        rel_info = f", rel={node.leftRel}" if node.leftRel else ""
        print(f"{indent}{node} (order={node.wordOrder}{rel_info})")
        
        for child in node.children:
            self.print_tree(child, level + 1)

    def delete_node(self, node: ParseTreeNode) -> None:
        """Remove specified node from tree and connect its children to parent"""
        if node not in self.nodes:
            return  # Node not in tree, return directly
        
        # 1. Connect children to parent
        if node.parent and node.children:
            for child in node.children:
                child.parent = node.parent
                if child not in node.parent.children:
                    node.parent.children.append(child)
        
        # 2. Remove from parent's children list
        if node.parent and node in node.parent.children:
            node.parent.children.remove(node)
        
        # 3. Remove from nodes list
        self.nodes.remove(node)
        
        # 4. Special handling: if deleting root node, try to find new root
        if node == self.root:
            # Select first node without parent as new root
            self.root = next((n for n in self.nodes if n.parent is None), None)
            if not self.root and self.nodes:
                # If no such node, set first node as root with parent=None
                self.root = self.nodes[0]
                self.root.parent = None

class StanfordNLParser:
    """Natural language parser that generates syntax parse trees"""
    def __init__(self):
        # Initialize stanza parser
        self.nlp = stanza.Pipeline(
            lang='en', 
            processors='tokenize,pos,lemma,depparse',
            tokenize_pretokenized=True  # Assume input is already tokenized
        )

    def parse(self, query: Query) -> None:
        """Parse query and build syntax parse tree"""
        self._stanford_parse(query)
        self._build_tree(query)
        self._fix_conj(query)

    def _stanford_parse(self, query: Query) -> None:
        """Perform syntax parsing using Stanford Parser"""
        # Convert tokenized results to format processable by stanza
        doc = self.nlp([query.sentence["question_tokens"]])
        
        # Process dependency parsing results
        tree_table = []  # Store tree table entries
        conj_table = []  # Store coordination relations
        
        # Assume single sentence
        sentence = doc.sentences[0]
        
        # Build tree table
        for dep in sentence.dependencies:
            # Dependency format: (governor, relation, dependent)
            governor = dep[0]
            relation = dep[1]
            dependent = dep[2]
            
            # Note: stanza indices start at 1, 0 represents root
            dep_index = dependent.id
            dep_value = dependent.text
            pos = dependent.xpos  # Use XPOS (Penn Treebank tags)
            gov_index = governor.id
            
            # Build tree table entry
            tree_table_entry = (str(dep_index), dep_value, pos, str(gov_index), relation)
            tree_table.append(tree_table_entry)
            
            # Handle coordination relations
            if relation.startswith('conj'):
                conj_entry = f"{gov_index} {dep_index}"
                conj_table.append(conj_entry)
        
        # Store results in query object
        query.treeTable = tree_table
        query.conjTable = conj_table

    def _build_tree(self, query: Query) -> None:
        """Build syntax parse tree from tree table"""
        # Modification: use query.parse_tree instead of query.parseTree
        query.parse_tree = ParseTree()
        
        # Mark processed entries
        done_list = [False] * len(query.treeTable)
        
        # First process root node
        for i, entry in enumerate(query.treeTable):
            if entry[3] == "0":  # Parent is root (0)
                query.parse_tree.build_node(entry)
                done_list[i] = True
        
        # Process remaining nodes until all are processed
        while not all(done_list):
            progress = False
            for i, entry in enumerate(query.treeTable):
                if not done_list[i]:
                    if query.parse_tree.build_node(entry):
                        done_list[i] = True
                        progress = True
                        break
            
            # If no progress in loop, indicates issue
            if not progress:
                break

    def _fix_conj(self, query: Query) -> None:
        """Fix coordination relations, set leftRel attribute"""
        if not query.conjTable:
            return
        
        for conj in query.conjTable:
            gov_num, dep_num = map(int, conj.split())
            gov_node = query.parse_tree.search_node_by_order(gov_num)
            dep_node = query.parse_tree.search_node_by_order(dep_num)
            
            if not gov_node or not dep_node:
                continue
            
            # Determine coordination logic word
            logic = ","
            prev_node = query.parse_tree.search_node_by_order(dep_node.wordOrder - 1)
            if prev_node:
                logic = prev_node.label.lower()
            
            # Set coordination relation marker
            if logic == "or":
                dep_node.leftRel = "or"
                # Check if gov_node.parent exists to avoid None error
                if gov_node.parent:
                    for sibling in gov_node.parent.children:
                        if sibling.leftRel == ",":
                            sibling.leftRel = "or"
            elif logic in ("and", "but"):
                dep_node.leftRel = "and"
                # Check if gov_node.parent exists
                if gov_node.parent:
                    for sibling in gov_node.parent.children:
                        if sibling.leftRel == ",":
                            sibling.leftRel = "and"
            else:
                dep_node.leftRel = ","
            
            # Adjust tree structure: move dep_node to same level as gov_node
            # First check if dep_node is among gov_node's children
            if dep_node in gov_node.children:
                gov_node.children.remove(dep_node)
            
            # Critical fix: ensure parent exists before operation
            if gov_node.parent is not None:
                dep_node.parent = gov_node.parent
                # Check if dep_node already in parent's children to avoid duplicates
                if dep_node not in dep_node.parent.children:
                    dep_node.parent.children.append(dep_node)
            else:
                # If gov_node is root (no parent), make dep_node child of root
                dep_node.parent = gov_node
                if dep_node not in dep_node.parent.children:
                    dep_node.parent.children.append(dep_node)
            
            # Inherit relation type
            dep_node.relationship = gov_node.relationship

# Example usage
if __name__ == "__main__":
    # Assume we have a Query object
    # Here's a simple example creation
    from query import SchemaGraph  # Assume SchemaGraph is in another file
    
    # Load database schema info (example with empty data)
    db_info = {
        "tables": [["stadium"], ["singer"], ["concert"], ["singer", "in", "concert"]],
        "columns": [...]  # Omitting full column definition
    }
    schema_graph = SchemaGraph(db_info)
    
    # Create query object
    query = Query(
        raw_question="How many singers do we have?",
        question_tokens=["how", "many", "singer", "do", "we", "have", "?"],
        schema_graph=schema_graph
    )
    
    # Initialize parser and parse query
    parser = StanfordNLParser()
    parser.parse(query)
    
    # Print parse tree (for debugging)
    # Modification: use query.parse_tree instead of query.parseTree
    if query.parse_tree:
        print("Syntax parse tree structure:")
        query.parse_tree.print_tree()
    
    # Print tree table (for debugging)
    print("\nTree table contents:")
    for entry in query.treeTable:
        print(entry)