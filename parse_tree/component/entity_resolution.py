from typing import List, Optional
from .query import Query, SchemaElement
from .stanzaParser import ParseTreeNode

class EntityPair:
    """实体对，记录两个节点的关联关系"""
    def __init__(self, left_node: ParseTreeNode, right_node: ParseTreeNode):
        self.left_node = left_node  # 左节点
        self.right_node = right_node  # 右节点
        self.relation = self._infer_relation()  # 推断关系类型

    def _infer_relation(self) -> str:
        """推断实体对的关系类型"""
        # 值节点与实体节点的关联（VTTEXT - NT）
        if (self.left_node.token_type == "VTTEXT" and self.right_node.token_type == "NT") or \
           (self.left_node.token_type == "NT" and self.right_node.token_type == "VTTEXT"):
            return "value_to_entity"
        
        # 实体节点之间的关联（NT - NT）
        elif self.left_node.token_type == "NT" and self.right_node.token_type == "NT":
            return "entity_to_entity"
        
        # 重复值节点的关联（VTTEXT - VTTEXT）
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
        """实体解析主函数：识别实体对并存储到query.entities"""
        print("\n----- 步骤7: 实体解析 (Entity Resolution) -----")
        if not query.parse_tree or not query.parse_tree.nodes:
            print("解析树为空，无法进行实体解析")
            return
        
        # 初始化entities列表
        query.entities = []
        nodes = query.parse_tree.nodes  # 获取所有节点
        
        # 遍历所有节点对，识别符合规则的实体对
        for i in range(len(nodes)):
            left_node = nodes[i]
            left_map = EntityResolution._get_best_mapped_schema(left_node)
            if not left_map:
                continue  # 跳过无有效映射的节点
            
            for j in range(i + 1, len(nodes)):
                right_node = nodes[j]
                right_map = EntityResolution._get_best_mapped_schema(right_node)
                if not right_map:
                    continue  # 跳过无有效映射的节点
                
                # 规则1-3：两个节点必须映射到同一数据库实体
                if EntityResolution._is_same_schema(left_map, right_map):
                    # 检查节点类型组合是否符合规则
                    if EntityResolution._is_valid_node_type_combination(left_node, right_node):
                        # 检查位置距离（值节点-实体节点/实体节点-实体节点需要距离≤2）
                        if EntityResolution._is_position_close(left_node, right_node, left_node.token_type, right_node.token_type):
                            # 创建实体对并添加到结果
                            entity_pair = EntityPair(left_node, right_node)
                            query.entities.append(entity_pair)
                            print(f"识别实体对: {entity_pair}")
        
        # 打印解析结果
        print(f"共识别 {len(query.entities)} 个实体对")
        if query.entities:
            print("实体解析结果:")
            for idx, pair in enumerate(query.entities, 1):
                print(f"  {idx}. {pair}")

    @staticmethod
    def _get_best_mapped_schema(node: ParseTreeNode) -> Optional[SchemaElement]:
        """获取节点最佳匹配的数据库实体（取相似度最高的映射）"""
        if hasattr(node, 'mapped_elements') and node.mapped_elements:
            # 选择相似度最高的映射（已排序）
            return node.mapped_elements[0].schema_element
        return None

    @staticmethod
    def _is_same_schema(left: SchemaElement, right: SchemaElement) -> bool:
        """判断两个映射是否指向同一数据库实体（表或字段）"""
        # 比较实体ID和名称（确保完全一致）
        return (left.element_id == right.element_id and 
                left.name == right.name and 
                left.type == right.type)

    @staticmethod
    def _is_valid_node_type_combination(left: ParseTreeNode, right: ParseTreeNode) -> bool:
        """检查节点类型组合是否符合规则（值节点-实体节点/实体节点-实体节点/值节点-值节点）"""
        lt, rt = left.token_type, right.token_type
        # 允许的类型组合：(VTTEXT, NT), (NT, VTTEXT), (NT, NT), (VTTEXT, VTTEXT)
        return (lt == "VTTEXT" and rt == "NT") or \
               (lt == "NT" and rt == "VTTEXT") or \
               (lt == "NT" and rt == "NT") or \
               (lt == "VTTEXT" and rt == "VTTEXT")

    @staticmethod
    def _is_position_close(left: ParseTreeNode, right: ParseTreeNode, lt: str, rt: str) -> bool:
        """检查两个节点的位置是否接近（距离≤2）"""
        distance = abs(left.wordOrder - right.wordOrder)
        
        # 规则：值节点-值节点无需距离限制，其他组合需距离≤2
        if (lt == "VTTEXT" and rt == "VTTEXT"):
            return True  # 规则3：重复值节点无距离限制
        else:
            return distance <= 2  # 规则1-2：距离≤2

    @staticmethod
    def _is_same_value(left: ParseTreeNode, right: ParseTreeNode) -> bool:
        """检查两个值节点的文本是否相同（规则3）"""
        return left.label.lower() == right.label.lower()