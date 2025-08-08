import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from .query import Query, SchemaGraph, SchemaElement
from .stanzaParser import ParseTree, ParseTreeNode
import string
import math
from collections import deque

# Similarity calculation functions
class SimFunctions:
    @staticmethod
    def similarity(node: ParseTreeNode, mapped_element: 'MappedSchemaElement') -> float:
        """Calculate similarity between node and mapped element"""
        # Character similarity
        char_sim = SimFunctions.char_similarity(node.label, mapped_element.schema_element.name)
        
        # POS tag similarity
        pos_sim = SimFunctions.pos_similarity(node.pos, mapped_element.schema_element)
        
        # Combined similarity
        total_sim = (char_sim * 0.6) + (pos_sim * 0.4)
        mapped_element.similarity = total_sim
        return total_sim
    
    @staticmethod
    def char_similarity(s1: str, s2: str) -> float:
        """Calculate string similarity (simplified version)"""
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        
        if s1 == s2:
            return 1.0
            
        # Calculate longest common subsequence length
        lcs_length = SimFunctions.lcs(s1, s2)
        return lcs_length / max(len(s1), len(s2))
    
    @staticmethod
    def lcs(s1: str, s2: str) -> int:
        """Longest common subsequence length"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    @staticmethod
    def pos_similarity(pos: str, schema_element: SchemaElement) -> float:
        """POS tag matching degree"""
        # For tables, noun POS tags match better
        if schema_element.type == "table" and pos.startswith("NN"):
            return 1.0
        # For columns, noun POS tags match better
        if schema_element.type == "column" and pos.startswith("NN"):
            return 0.9
        return 0.5
    
    @staticmethod
    def lemmatize(word: str) -> str:
        """Simple lemmatization (in practice, use libraries like NLTK)"""
        # Simplified implementation
        if word.endswith("s") and len(word) > 1:
            return word[:-1]
        return word

# Mapped schema element class
class MappedSchemaElement:
    def __init__(self, schema_element: SchemaElement):
        self.schema_element = schema_element  # Mapped schema element
        self.similarity = 0.0  # Similarity score
        self.mapped_values = []  # Mapped values
        self.choice = 0  # Selection marker
        self.score = 0.0  # Score

    def __repr__(self) -> str:
        return f"MappedSchemaElement({self.schema_element.name}, similarity={self.similarity:.2f})"

    def __lt__(self, other: 'MappedSchemaElement') -> bool:
        # For sorting, higher similarity comes first
        return self.similarity > other.similarity

class NodeMapper:
    @staticmethod
    def phrase_process(query: Query, tokens_path: str) -> None:
        """Main phrase mapping process"""
        print("\n===== Starting phrase mapping process =====")
        # Parse tokens.xml file
        try:
            tokens_tree = ET.parse(tokens_path)
            tokens_root = tokens_tree.getroot()
            print(f"Successfully loaded tokens file: {tokens_path}")
        except Exception as e:
            print(f"Failed to load tokens file: {e}")
            return
        
        # Execute each step
        NodeMapper.tokenize(query, tokens_root)
        NodeMapper.delete_useless(query)
        NodeMapper.map_nodes(query)
        NodeMapper.delete_no_match(query)

        print("\n----- Optimized parse tree structure (before sorting) -----")
        if query.parse_tree and query.parse_tree.root:
            query.parse_tree.print_tree()  # Call ParseTree's print method
        else:
            print("Parse tree is empty or missing root node")

        NodeMapper.individual_ranking(query)
        NodeMapper.group_ranking(query)
        print("===== Phrase mapping process completed =====")
    
    @staticmethod
    def tokenize(query: Query, tokens_root: ET.Element) -> None:
        """Node type annotation"""
        print("\n----- Step 1: Node type annotation (tokenize) -----")
        parse_tree = query.parse_tree
        if not parse_tree or not parse_tree.root:
            print("Parse tree is empty, cannot annotate")
            return
            
        # Mark root node
        parse_tree.root.token_type = "ROOT"
        print(f"Root node marked: {parse_tree.root.label} (type: {parse_tree.root.token_type})")
        
        # Mark core verbs (CMT)
        cmt_count = 0
        for child in parse_tree.root.children:
            if NodeMapper.is_of_type(tokens_root, parse_tree, child, "CMT", None):
                child.token_type = "CMT"
                cmt_count += 1
                print(f"Core verb marked: {child.label} (type: {child.token_type})")
        print(f"Total {cmt_count} core verbs (CMT) marked")
        
        # Mark negation words (NEG)
        neg_count = 0
        for node in parse_tree.nodes:
            if not hasattr(node, 'token_type') or node.token_type == "NA":
                node.token_type = "NA"  # Initialize unrecognized type
                
            if node.token_type == "NA" and NodeMapper.is_of_type(tokens_root, parse_tree, node, "NEG", None):
                node.token_type = "NEG"
                neg_count += 1
                print(f"Negation word marked: {node.label} (type: {node.token_type})")
        print(f"Total {neg_count} negation words (NEG) marked")
        
        # Merge multi-word expressions
        original_node_count = len(parse_tree.nodes)
        NodeMapper.merge_multi_word_expressions(parse_tree)
        merged_count = original_node_count - len(parse_tree.nodes)
        print(f"Merged multi-word expressions: {merged_count} nodes merged")
        
        # Other type annotations
        current_size = 0
        type_counts = {"FT":0, "OT":0, "QT":0, "VT":0, "NTVT":0, "JJ":0, "NA":0}
        
        while current_size != len(parse_tree.nodes):
            current_size = len(parse_tree.nodes)
            for node in parse_tree.nodes[:]:  # Use copy to avoid modification errors
                if node.token_type == "NA":
                    # Function nodes (FT)
                    if NodeMapper.is_of_type(tokens_root, parse_tree, node, "FT", "function"):
                        node.token_type = "FT"
                        type_counts["FT"] += 1
                    # Operator nodes (OT)
                    elif NodeMapper.is_of_type(tokens_root, parse_tree, node, "OT", "operator"):
                        node.token_type = "OT"
                        type_counts["OT"] += 1
                    # Quantity nodes (QT)
                    elif NodeMapper.is_of_type(tokens_root, parse_tree, node, "QT", "quantity"):
                        node.token_type = "QT"
                        type_counts["QT"] += 1
                    # Numeric nodes (VT)
                    elif NodeMapper.is_numeric(node.label):
                        node.token_type = "VT"
                        type_counts["VT"] += 1
                    # Nominal nodes (NTVT)
                    elif node.pos.startswith("NN") or node.pos == "CD":
                        node.token_type = "NTVT"
                        type_counts["NTVT"] += 1
                    # Adjective nodes (JJ)
                    elif node.pos.startswith("JJ"):
                        node.token_type = "JJ"
                        type_counts["JJ"] += 1
                    else:
                        type_counts["NA"] += 1
        
        # Print type statistics
        print("\nNode type statistics:")
        for type_name, count in type_counts.items():
            if count > 0:
                print(f"  {type_name}: {count} nodes")
        
        # Print all node types
        print("\nAll node type annotation results:")
        for node in parse_tree.nodes:  # Only print first 10 to avoid being too long
            print(f"  {node.label} (POS: {node.pos}, Type: {node.token_type})")
        # if len(parse_tree.nodes) > 10:
        #     print(f"  ... Total {len(parse_tree.nodes)} nodes")
    
    @staticmethod
    def merge_multi_word_expressions(parse_tree: ParseTree) -> None:
        """Merge multi-word phrases"""
        for node in parse_tree.nodes[:]:  # Use copy to avoid modification errors
            if hasattr(node, 'relationship') and node.relationship == "mwe" and node.token_type == "NA":
                original_parent_label = node.parent.label
                if node.wordOrder > node.parent.wordOrder:
                    node.parent.label += " " + node.label
                else:
                    node.parent.label = node.label + " " + node.parent.label
                print(f"Merged phrase: {node.label} -> Parent node becomes: {node.parent.label}")
                parse_tree.delete_node(node)
    
    @staticmethod
    def delete_useless(query: Query) -> None:
        """Remove irrelevant nodes"""
        print("\n----- Step 2: Remove irrelevant nodes (deleteUseless) -----")
        parse_tree = query.parse_tree
        if not parse_tree:
            print("Parse tree is empty, cannot clean")
            return
            
        original_node_count = len(parse_tree.nodes)
        print(f"Node count before cleaning: {original_node_count}")
        
        # Save original parse tree
        query.original_parse_tree = ParseTree()  # Simplified handling
        query.original_parse_tree.root = parse_tree.root
        
        # Clean NA and QT type nodes
        deleted_nodes = []
        prepositions = []
        for node in parse_tree.nodes[:]:  # Use copy to avoid modification errors
            if hasattr(node, 'token_type') and (node.token_type == "NA" or node.token_type == "QT"):
                # Special handling for prepositions
                if node.label.lower() in ["on", "in", "of", "by", "for", "with"]:
                    prep_info = f"Preposition {node.label} info saved"
                    if node.children:
                        node.children[0].prep = node.label
                        prep_info += f" to child node {node.children[0].label}"
                    elif node.parent:
                        node.parent.prep = node.label
                        prep_info += f" to parent node {node.parent.label}"
                    prepositions.append(prep_info)
                
                deleted_nodes.append(node.label)
                parse_tree.delete_node(node)
        
        # Print cleaning results
        print(f"Node count after cleaning: {len(parse_tree.nodes)}")
        print(f"Total {original_node_count - len(parse_tree.nodes)} nodes deleted")
        print(f"Deleted nodes: {deleted_nodes[:5]} {'...' if len(deleted_nodes) > 5 else ''}")
        if prepositions:
            print("Processed prepositions:")
            for prep in prepositions[:3]:
                print(f"  {prep}")
    
    @staticmethod
    def map_nodes(query: Query) -> None:
        """Map nodes to database schema elements"""
        print("\n----- Step 3: Node to database schema mapping (map) -----")
        parse_tree = query.parse_tree
        if not parse_tree:
            print("Parse tree is empty, cannot map")
            return
            
        # Initialize mapped_elements attribute
        for node in parse_tree.nodes:
            node.mapped_elements = []
        
        # Map nominal and adjective nodes
        mapped_count = 0
        for node in parse_tree.nodes:
            if node.token_type in ["NTVT", "JJ"]:
                # Check if exists in schema
                NodeMapper.map_schema_element(node, query.graph)
                # If there are mappings
                if node.mapped_elements:
                    mapped_count += 1
                    top_match = node.mapped_elements[0]
                    print(f"Node {node.label} mapped to: {top_match.schema_element.name} (similarity: {top_match.similarity:.2f})")
                else:
                    node.token_type = "NA"
                    print(f"Node {node.label} has no matching database element, marked as NA")
            
            # Map numeric nodes
            elif node.token_type == "VT":
                # Determine operator
                ot = "="
                if hasattr(node.parent, 'token_type') and node.parent.token_type == "OT":
                    ot = node.parent.label
                elif node.children:
                    for child in node.children:
                        if hasattr(child, 'token_type') and child.token_type == "OT":
                            ot = child.label
                            if ot == "NA" and child.label.lower() == "at least":
                                ot = ">="
                
                # Map numeric value
                NodeMapper.map_numeric_element(node, ot)
                node.token_type = "VTNUM"
                print(f"Numeric node {node.label} mapped (operator: {ot})")
                mapped_count += 1
        
        print(f"Successfully mapped nodes: {mapped_count}")
    
    @staticmethod
    def map_schema_element(node: ParseTreeNode, schema_graph: SchemaGraph) -> None:
        """Map node to database schema elements"""
        # Check matches with tables and columns
        for elem in schema_graph.schema_elements:
            # Calculate similarity
            mapped_elem = MappedSchemaElement(elem)
            SimFunctions.similarity(node, mapped_elem)
            
            # Only keep if similarity is high enough
            if mapped_elem.similarity > 0.3:  # Threshold can be adjusted
                node.mapped_elements.append(mapped_elem)
        
        # Sort mapping results
        node.mapped_elements.sort()
    
    @staticmethod
    def map_numeric_element(node: ParseTreeNode, operator: str) -> None:
        """Map numeric elements"""
        # Create numeric mapping
        mapped_elem = MappedSchemaElement(SchemaElement(-1, f"VALUE:{node.label}", "value"))
        mapped_elem.operator = operator
        mapped_elem.mapped_values = [node.label]
        mapped_elem.similarity = 1.0  # Numeric match score is 1
        node.mapped_elements.append(mapped_elem)
    
    @staticmethod
    def delete_no_match(query: Query) -> None:
        """Remove nodes with no matches"""
        print("\n----- Step 4: Remove unmatched nodes (deleteNoMatch) -----")
        parse_tree = query.parse_tree
        if not parse_tree:
            print("Parse tree is empty, cannot delete")
            return
            
        original_node_count = len(parse_tree.nodes)
        print(f"Node count before deletion: {original_node_count}")
        
        deleted_nodes = []
        for node in parse_tree.nodes[:]:  # Use copy to avoid modification errors
            if hasattr(node, 'token_type') and node.token_type == "NA":
                # Special handling for prepositions
                if node.label.lower() in ["on", "in"] and node.parent:
                    node.parent.prep = node.label
                    print(f"Preposition {node.label} info saved to parent node {node.parent.label}")
                
                deleted_nodes.append(node.label)
                parse_tree.delete_node(node)
        
        print(f"Node count after deletion: {len(parse_tree.nodes)}")
        print(f"Total {original_node_count - len(parse_tree.nodes)} unmatched nodes deleted")
        print(f"Deleted nodes: {deleted_nodes}")
    
    @staticmethod
    def individual_ranking(query: Query) -> None:
        """Single node mapping ranking"""
        print("\n----- Step 5: Single node mapping ranking (individualRanking) -----")
        parse_tree = query.parse_tree
        if not parse_tree:
            print("Parse tree is empty, cannot rank")
            return
            
        ranked_nodes = 0
        for node in parse_tree.nodes:
            if hasattr(node, 'mapped_elements') and node.mapped_elements:
                # Ensure similarity is calculated
                for mapped_elem in node.mapped_elements:
                    SimFunctions.similarity(node, mapped_elem)
                
                # Sort
                node.mapped_elements.sort()
                ranked_nodes += 1
                
                # Print ranking results
                print(f"\nNode {node.label} mapping ranking:")
                for i, elem in enumerate(node.mapped_elements[:3]):  # Only print top 3
                    print(f"  Rank {i+1}: {elem.schema_element.name} (similarity: {elem.similarity:.4f})")
                if len(node.mapped_elements) > 3:
                    print(f"  ... Total {len(node.mapped_elements)} mappings")
                
                # Handle duplicate mappings
                if node.token_type == "NTVT":
                    original_count = len(node.mapped_elements)
                    NodeMapper.handle_duplicate_mappings(node)
                    if len(node.mapped_elements) < original_count:
                        print(f"  After deduplication: {len(node.mapped_elements)} mappings kept")
        
        print(f"Completed ranking for {ranked_nodes} nodes")
    
    @staticmethod
    def handle_duplicate_mappings(node: ParseTreeNode) -> None:
        """Handle duplicate mappings"""
        delete_list = []
        mapped_elements = node.mapped_elements
        
        for j in range(len(mapped_elements)):
            nt = mapped_elements[j]
            for k in range(j + 1, len(mapped_elements)):
                vt = mapped_elements[k]
                # If different mappings of the same schema element
                if (not nt.mapped_values and vt.mapped_values and 
                    nt.schema_element.element_id == vt.schema_element.element_id):
                    if nt.similarity >= vt.similarity:
                        vt.similarity = nt.similarity
                        vt.choice = -1
                        # Swap positions
                        mapped_elements[j], mapped_elements[k] = mapped_elements[k], mapped_elements[j]
                    delete_list.append(nt)
        
        # Remove duplicates
        for elem in delete_list:
            if elem in mapped_elements:
                mapped_elements.remove(elem)
    
    @staticmethod
    def group_ranking(query: Query) -> None:
        """Global mapping optimization"""
        print("\n----- Step 6: Global mapping optimization (groupRanking) -----")
        parse_tree = query.parse_tree
        if not parse_tree or not parse_tree.nodes:
            print("Parse tree is empty, cannot optimize")
            return
            
        # Find best root node (node with most unambiguous mapping)
        root = parse_tree.nodes[0]
        root_score = 0
        
        for node in parse_tree.nodes:
            if hasattr(node, 'mapped_elements') and node.mapped_elements:
                # Calculate mapping clarity score
                if len(node.mapped_elements) == 1:
                    score = 1.0
                else:
                    # Higher score means bigger advantage of 1st place over 2nd
                    try:
                        score = 1 - node.mapped_elements[1].similarity / node.mapped_elements[0].similarity
                    except (IndexError, ZeroDivisionError):
                        score = 0
                
                if score > root_score:
                    root = node
                    root_score = score
        
        print(f"Best root node: {root.label} (clarity score: {root_score:.4f})")
        if root.label == "ROOT":
            print("Root node is ROOT, no further optimization needed")
            return
            
        # Select best mapping
        root.choice = 0
        print(f"Root node {root.label} selected best mapping: {root.mapped_elements[0].schema_element.name}")
        
        # BFS traversal for global optimization
        done = {node: False for node in parse_tree.nodes}
        queue = deque()
        queue.append((root, root))  # (parent, child)
        optimized_count = 0
        
        while queue:
            parent, child = queue.popleft()
            
            if not done.get(child, True):
                if parent != child and hasattr(child, 'mapped_elements') and child.mapped_elements:
                    # Calculate best mapping
                    max_score = -1
                    max_position = 0
                    
                    for i, child_elem in enumerate(child.mapped_elements):
                        if not parent.mapped_elements or parent.choice >= len(parent.mapped_elements):
                            continue
                            
                        parent_elem = parent.mapped_elements[parent.choice]
                        # Calculate distance
                        try:
                            distance = parent.graph.shortest_distance[
                                parent_elem.schema_element.element_id][
                                child_elem.schema_element.element_id
                            ]
                        except:
                            distance = 1.0
                        
                        # Calculate score
                        score = parent_elem.similarity * child_elem.similarity * distance
                        
                        if score > max_score:
                            max_score = score
                            max_position = i
                    
                    child.choice = max_position
                    optimized_count += 1
                    print(f"Node {child.label} best mapping: {child.mapped_elements[max_position].schema_element.name} (score: {max_score:.4f})")
                
                # Add child and parent nodes to queue
                for grandchild in child.children:
                    queue.append((child, grandchild))
                
                if child.parent and child.parent != child:
                    queue.append((child, child.parent))
                
                done[child] = True
        
        # Update node types
        nt_count = 0
        vttext_count = 0
        for node in parse_tree.nodes:
            if node.token_type in ["NTVT", "JJ"] and hasattr(node, 'mapped_elements') and node.mapped_elements:
                if (node.choice < len(node.mapped_elements) and 
                    (not node.mapped_elements[node.choice].mapped_values or 
                     node.mapped_elements[node.choice].choice == -1)):
                    node.token_type = "NT"  # Entity node
                    nt_count += 1
                else:
                    node.token_type = "VTTEXT"  # Value node
                    vttext_count += 1
        
        print(f"Optimization completed, processed {optimized_count} nodes")
        print(f"Entity nodes (NT): {nt_count}, Value nodes (VTTEXT): {vttext_count}")
    
    @staticmethod
    def is_of_type(tokens_root: ET.Element, parse_tree: ParseTree, node: ParseTreeNode, token_type: str, tag: Optional[str]) -> bool:
        """Check if node belongs to specific type"""
        # Check two matching patterns
        if NodeMapper._is_of_type(tokens_root, parse_tree, node, token_type, 1, tag):
            return True
        if NodeMapper._is_of_type(tokens_root, parse_tree, node, token_type, 2, tag):
            return True
        return False
    
    @staticmethod
    def _is_of_type(tokens_root: ET.Element, parse_tree: ParseTree, node: ParseTreeNode, 
                   token_type: str, case_type: int, tag: Optional[str]) -> bool:
        """Internal implementation for checking node type"""
        # Get label text
        if case_type == 1:
            label = node.label.lower()
        else:
            label = SimFunctions.lemmatize(node.label).lower()
        
        # Find corresponding type in tokens.xml
        token_elements = tokens_root.findall(f".//{token_type}")
        if not token_elements:
            return False
            
        for token_elem in token_elements:
            phrase_elements = token_elem.findall("phrase")
            for phrase_elem in phrase_elements:
                phrase_text = phrase_elem.text.strip().lower() if phrase_elem.text else ""
                
                # Single word matching
                if len(phrase_text.split()) == 1 and " " not in label:
                    if label == phrase_text:
                        if tag and phrase_elem.find(tag) is not None:
                            node.function = phrase_elem.find(tag).text.strip()
                        return True
                
                # Phrase containment matching
                elif phrase_text in label:
                    if tag and phrase_elem.find(tag) is not None:
                        node.function = phrase_elem.find(tag).text.strip()
                    return True
                    
        return False
    
    @staticmethod
    def is_numeric(s: str) -> bool:
        """Check if string is numeric"""
        s = s.strip()
        if not s:
            return False
            
        # Check if integer or float
        if s.replace('.', '', 1).isdigit():
            return True
            
        # Check if signed number
        if (s[0] in '+-' and len(s) > 1 and 
            s[1:].replace('.', '', 1).isdigit()):
            return True
            
        return False

# Enhance ParseTreeNode class with required attributes and methods
def enhance_parse_tree_node():
    """Enhance ParseTreeNode class with required attributes"""
    # Add token_type attribute
    if not hasattr(ParseTreeNode, 'token_type'):
        ParseTreeNode.token_type = "NA"
    
    # Add mapped_elements attribute
    if not hasattr(ParseTreeNode, 'mapped_elements'):
        ParseTreeNode.mapped_elements = []
    
    # Add prep attribute
    if not hasattr(ParseTreeNode, 'prep'):
        ParseTreeNode.prep = None
    
    # Add choice attribute
    if not hasattr(ParseTreeNode, 'choice'):
        ParseTreeNode.choice = 0

# Enhance ParseTreeNode class
enhance_parse_tree_node()