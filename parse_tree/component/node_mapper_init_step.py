import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from .query import Query, SchemaGraph, SchemaElement
from .stanzaParser import ParseTree, ParseTreeNode
import string
import math
from collections import deque

# Similarity calculation functions (kept, but not called in later steps)
class SimFunctions:
    @staticmethod
    def similarity(node: ParseTreeNode, mapped_element: 'MappedSchemaElement') -> float:
        """Calculate similarity between a node and a mapped element"""
        char_sim = SimFunctions.char_similarity(node.label, mapped_element.schema_element.name)
        pos_sim = SimFunctions.pos_similarity(node.pos, mapped_element.schema_element)
        total_sim = (char_sim * 0.6) + (pos_sim * 0.4)
        mapped_element.similarity = total_sim
        return total_sim
    
    @staticmethod
    def char_similarity(s1: str, s2: str) -> float:
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        if s1 == s2:
            return 1.0
        lcs_length = SimFunctions.lcs(s1, s2)
        return lcs_length / max(len(s1), len(s2))
    
    @staticmethod
    def lcs(s1: str, s2: str) -> int:
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
        if schema_element.type == "table" and pos.startswith("NN"):
            return 1.0
        if schema_element.type == "column" and pos.startswith("NN"):
            return 0.9
        return 0.5
    
    @staticmethod
    def lemmatize(word: str) -> str:
        if word.endswith("s") and len(word) > 1:
            return word[:-1]
        return word

# Mapped schema element class (kept, but not called in later steps)
class MappedSchemaElement:
    def __init__(self, schema_element: SchemaElement):
        self.schema_element = schema_element
        self.similarity = 0.0
        self.mapped_values = []
        self.choice = 0
        self.score = 0.0

    def __repr__(self) -> str:
        return f"MappedSchemaElement({self.schema_element.name}, similarity={self.similarity:.2f})"

    def __lt__(self, other: 'MappedSchemaElement') -> bool:
        return self.similarity > other.similarity

class NodeMapper:
    @staticmethod
    def phrase_process(query: Query, tokens_path: str) -> List[Tuple[str, int]]:
        """Phrase mapping process: returns a list of (remaining word, original position id)"""
        parse_tree = query.parse_tree
        if not parse_tree or not parse_tree.root:
            print("Parse tree is empty, cannot process")
            return []

        try:
            tokens_tree = ET.parse(tokens_path)
            tokens_root = tokens_tree.getroot()
        except Exception as e:
            print(f"Failed to load tokens file: {e}")
            return []

        NodeMapper.tokenize(query, tokens_root)
        remaining_nodes = NodeMapper.delete_useless(query)  # Format: [(word, original position id), ...]
        
        return remaining_nodes  # Return the result
        
    
    @staticmethod
    def tokenize(query: Query, tokens_root: ET.Element) -> None:
        """Node type annotation (Step 1)"""
        parse_tree = query.parse_tree
        if not parse_tree or not parse_tree.root:
            return
            
        # Mark root node
        parse_tree.root.token_type = "ROOT"
        
        # Mark core verb (CMT)
        cmt_count = 0
        for child in parse_tree.root.children:
            if NodeMapper.is_of_type(tokens_root, parse_tree, child, "CMT", None):
                child.token_type = "CMT"
                cmt_count += 1
        
        # Mark negation words (NEG)
        neg_count = 0
        for node in parse_tree.nodes:
            if not hasattr(node, 'token_type') or node.token_type == "NA":
                node.token_type = "NA"  # Initialize unrecognized type
                
            if node.token_type == "NA" and NodeMapper.is_of_type(tokens_root, parse_tree, node, "NEG", None):
                node.token_type = "NEG"
                neg_count += 1
        
        # Merge multi-word expressions
        original_node_count = len(parse_tree.nodes)
        NodeMapper.merge_multi_word_expressions(parse_tree)
        merged_count = original_node_count - len(parse_tree.nodes)
        
        # Other type annotations
        current_size = 0
        type_counts = {"FT":0, "OT":0, "QT":0, "VT":0, "NTVT":0, "JJ":0, "NA":0}
        
        while current_size != len(parse_tree.nodes):
            current_size = len(parse_tree.nodes)
            for node in parse_tree.nodes[:]:
                if node.token_type == "NA":
                    # Function node (FT)
                    if NodeMapper.is_of_type(tokens_root, parse_tree, node, "FT", "function"):
                        node.token_type = "FT"
                        type_counts["FT"] += 1
                    # Operator node (OT)
                    elif NodeMapper.is_of_type(tokens_root, parse_tree, node, "OT", "operator"):
                        node.token_type = "OT"
                        type_counts["OT"] += 1
                    # Quantity node (QT)
                    elif NodeMapper.is_of_type(tokens_root, parse_tree, node, "QT", "quantity"):
                        node.token_type = "QT"
                        type_counts["QT"] += 1
                    # Numeric value node (VT)
                    elif NodeMapper.is_numeric(node.label):
                        node.token_type = "VT"
                        type_counts["VT"] += 1
                    # Noun-type node (NTVT)
                    elif node.pos.startswith("NN") or node.pos == "CD":
                        node.token_type = "NTVT"
                        type_counts["NTVT"] += 1
                    # Adjective node (JJ)
                    elif node.pos.startswith("JJ"):
                        node.token_type = "JJ"
                        type_counts["JJ"] += 1
                    else:
                        type_counts["NA"] += 1
    
    @staticmethod
    def merge_multi_word_expressions(parse_tree: ParseTree) -> None:
        """Merge multi-word expressions (sub-operation in Step 1)"""
        for node in parse_tree.nodes[:]:
            if hasattr(node, 'relationship') and node.relationship == "mwe" and node.token_type == "NA":
                original_parent_label = node.parent.label
                if node.wordOrder > node.parent.wordOrder:
                    node.parent.label += " " + node.label
                else:
                    node.parent.label = node.label + " " + node.parent.label
                parse_tree.delete_node(node)
    
    @staticmethod
    def delete_useless(query: Query) -> List[Tuple[str, int]]:
        """Remove irrelevant nodes and return a list of (word, original position)"""
        parse_tree = query.parse_tree
        if not parse_tree:
            return []

        original_node_count = len(parse_tree.nodes)

        # Remove NA and QT type nodes
        deleted_nodes = []
        prepositions = []
        for node in parse_tree.nodes[:]:
            if hasattr(node, 'token_type') and (node.token_type == "NA" or node.token_type == "QT"):
                if node.label.lower() in ["on", "in", "of", "by", "for", "with"]:
                    prep_info = f"Preposition {node.label} information saved"
                    if node.children:
                        node.children[0].prep = node.label
                        prep_info += f" to child node {node.children[0].label}"
                    elif node.parent:
                        node.parent.prep = node.label
                        prep_info += f" to parent node {node.parent.label}"
                    prepositions.append(prep_info)
                deleted_nodes.append(node.label)
                parse_tree.delete_node(node)

        # Sort remaining nodes by original position and return (word, position)
        remaining_nodes = sorted(
            parse_tree.nodes,
            key=lambda x: x.wordOrder
        )
        result = [(node.label, node.wordOrder) for node in remaining_nodes]

        return result

    
    # The following methods are for later steps, but are no longer called; can be kept or removed
    @staticmethod
    def is_of_type(tokens_root: ET.Element, parse_tree: ParseTree, node: ParseTreeNode, token_type: str, tag: Optional[str]) -> bool:
        if NodeMapper._is_of_type(tokens_root, parse_tree, node, token_type, 1, tag):
            return True
        if NodeMapper._is_of_type(tokens_root, parse_tree, node, token_type, 2, tag):
            return True
        return False
    
    @staticmethod
    def _is_of_type(tokens_root: ET.Element, parse_tree: ParseTree, node: ParseTreeNode, 
                   token_type: str, case_type: int, tag: Optional[str]) -> bool:
        if case_type == 1:
            label = node.label.lower()
        else:
            label = SimFunctions.lemmatize(node.label).lower()
        token_elements = tokens_root.findall(f".//{token_type}")
        if not token_elements:
            return False
        for token_elem in token_elements:
            phrase_elements = token_elem.findall("phrase")
            for phrase_elem in phrase_elements:
                phrase_text = phrase_elem.text.strip().lower() if phrase_elem.text else ""
                if len(phrase_text.split()) == 1 and " " not in label and label == phrase_text:
                    if tag and phrase_elem.find(tag) is not None:
                        node.function = phrase_elem.find(tag).text.strip()
                    return True
                elif phrase_text in label:
                    if tag and phrase_elem.find(tag) is not None:
                        node.function = phrase_elem.find(tag).text.strip()
                    return True
        return False
    
    @staticmethod
    def is_numeric(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        if s.replace('.', '', 1).isdigit():
            return True
        if (s[0] in '+-' and len(s) > 1 and s[1:].replace('.', '', 1).isdigit()):
            return True
        return False

# Add required attributes and methods to ParseTreeNode
def enhance_parse_tree_node():
    if not hasattr(ParseTreeNode, 'token_type'):
        ParseTreeNode.token_type = "NA"
    if not hasattr(ParseTreeNode, 'prep'):
        ParseTreeNode.prep = None

enhance_parse_tree_node()
