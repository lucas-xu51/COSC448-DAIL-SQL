from typing import List, Optional
from .query import Query, SchemaElement
from .stanzaParser import ParseTreeNode

class EntityPair:
    """Entity pair, recording the relationship between two nodes"""
    def __init__(self, left_node: ParseTreeNode, right_node: ParseTreeNode):
        self.left_node = left_node  # Left node
        self.right_node = right_node  # Right node
        self.relation = self._infer_relation()  # Infer relationship type

    def _infer_relation(self) -> str:
        """Infer the relationship type of the entity pair"""
        # Relationship between a value node and an entity node (VTTEXT - NT)
        if (self.left_node.token_type == "VTTEXT" and self.right_node.token_type == "NT") or \
           (self.left_node.token_type == "NT" and self.right_node.token_type == "VTTEXT"):
            return "value_to_entity"
        
        # Relationship between two entity nodes (NT - NT)
        elif self.left_node.token_type == "NT" and self.right_node.token_type == "NT":
            return "entity_to_entity"
        
        # Relationship between duplicate value nodes (VTTEXT - VTTEXT)
        elif self.left_node.token_type == "VTTEXT" and self.right_node.token_type == "VTTEXT":
            return "value_to_value"
        
        return "unknown"

    def __repr__(self) -> str:
        return (f"EntityPair({self.left_node.label} [{self.left_node.token_type}] "
                f"→ {self.right_node.label} [{self.right_node.token_type}], "
                f"relation: {self.relation})")

class EntityResolution:
    @staticmethod
    def entity_resolute(query: Query) -> None:
        """Main entity resolution function: identify entity pairs and store them in query.entities"""
        print("\n----- Step 7: Entity Resolution -----")
        if not query.parse_tree or not query.parse_tree.nodes:
            print("Parse tree is empty, cannot perform entity resolution")
            return
        
        # Initialize entities list
        query.entities = []
        nodes = query.parse_tree.nodes  # Get all nodes
        
        # Iterate over all node pairs and identify entity pairs that match the rules
        for i in range(len(nodes)):
            left_node = nodes[i]
            left_map = EntityResolution._get_best_mapped_schema(left_node)
            if not left_map:
                continue  # Skip nodes without valid mapping
            
            for j in range(i + 1, len(nodes)):
                right_node = nodes[j]
                right_map = EntityResolution._get_best_mapped_schema(right_node)
                if not right_map:
                    continue  # Skip nodes without valid mapping
                
                # Rules 1-3: Both nodes must map to the same database entity
                if EntityResolution._is_same_schema(left_map, right_map):
                    # Check if node type combination matches the rules
                    if EntityResolution._is_valid_node_type_combination(left_node, right_node):
                        # Check position distance (value-to-entity/entity-to-entity requires distance ≤ 2)
                        if EntityResolution._is_position_close(left_node, right_node, left_node.token_type, right_node.token_type):
                            # Create entity pair and add to results
                            entity_pair = EntityPair(left_node, right_node)
                            query.entities.append(entity_pair)
                            print(f"Identified entity pair: {entity_pair}")
        
        # Print resolution results
        print(f"Identified {len(query.entities)} entity pairs in total")
        if query.entities:
            print("Entity resolution results:")
            for idx, pair in enumerate(query.entities, 1):
                print(f"  {idx}. {pair}")

    @staticmethod
    def _get_best_mapped_schema(node: ParseTreeNode) -> Optional[SchemaElement]:
        """Get the best-matched database entity for a node (highest similarity mapping)"""
        if hasattr(node, 'mapped_elements') and node.mapped_elements:
            # Select the mapping with the highest similarity (already sorted)
            return node.mapped_elements[0].schema_element
        return None

    @staticmethod
    def _is_same_schema(left: SchemaElement, right: SchemaElement) -> bool:
        """Check if two mappings point to the same database entity (table or column)"""
        # Compare entity ID and name (must be exactly the same)
        return (left.element_id == right.element_id and 
                left.name == right.name and 
                left.type == right.type)

    @staticmethod
    def _is_valid_node_type_combination(left: ParseTreeNode, right: ParseTreeNode) -> bool:
        """Check if the node type combination is valid (value-to-entity / entity-to-entity / value-to-value)"""
        lt, rt = left.token_type, right.token_type
        # Allowed combinations: (VTTEXT, NT), (NT, VTTEXT), (NT, NT), (VTTEXT, VTTEXT)
        return (lt == "VTTEXT" and rt == "NT") or \
               (lt == "NT" and rt == "VTTEXT") or \
               (lt == "NT" and rt == "NT") or \
               (lt == "VTTEXT" and rt == "VTTEXT")

    @staticmethod
    def _is_position_close(left: ParseTreeNode, right: ParseTreeNode, lt: str, rt: str) -> bool:
        """Check if two nodes are close in position (distance ≤ 2)"""
        distance = abs(left.wordOrder - right.wordOrder)
        
        # Rule: value-to-value nodes have no distance limit; other combinations require distance ≤ 2
        if (lt == "VTTEXT" and rt == "VTTEXT"):
            return True  # Rule 3: duplicate value nodes have no distance limit
        else:
            return distance <= 2  # Rules 1-2: distance ≤ 2

    @staticmethod
    def _is_same_value(left: ParseTreeNode, right: ParseTreeNode) -> bool:
        """Check if two value nodes have the same text (Rule 3)"""
        return left.label.lower() == right.label.lower()