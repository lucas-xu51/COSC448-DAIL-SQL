import copy
from typing import List, Dict, Optional, Tuple
from .query import Query, SchemaGraph  # 确保导入SchemaGraph
from .stanzaParser import ParseTree, ParseTreeNode

class Tree:
    """用于树结构调整的临时树类（扩展ParseTree功能）"""
    def __init__(self, parse_tree: ParseTree):
        self.all_nodes = [self._convert_node(n) for n in parse_tree.nodes]  # 转换为TreeNode
        self.root = self.all_nodes[0] if self.all_nodes else None
        self.cost = 0  # 树结构的成本（调整次数）
        self.invalid = 0  # 无效性评分（0为有效）
        self.weight = 0.0  # 树的权重（有效性评分）
        self.hash_num = 0  # 树的哈希值（用于去重）
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
            # 保存原始节点的子节点wordOrder（关键：用于后续建立父子关系）
            node.original_children = [child.wordOrder for child in parse_node.children]
            return node

    def _init_parents_and_children(self):
        node_map = {n.node_id: n for n in self.all_nodes}
        for node in self.all_nodes:
            # 为当前节点添加子节点（基于原始解析树的children信息）
            for child_id in node.original_children:
                child_node = node_map.get(child_id)
                if child_node:
                    node.children.append(child_node)
                    child_node.parent = node  # 同时设置子节点的父节点
        # 根节点是没有父节点的节点
        self.root = next((n for n in self.all_nodes if n.parent is None), None)

    def move_sub_tree(self, node: 'TreeNode', new_parent: 'TreeNode') -> bool:
        """将节点移动到新的父节点下"""
        if node == self.root:
            return False  # 根节点不能移动
        # 从原父节点移除
        if node.parent:
            node.parent.children.remove(node)
        # 添加到新父节点
        node.parent = new_parent
        new_parent.children.append(node)
        return True

    def add_equal(self):
        """为需要的节点添加等于操作符（处理max/min等函数的特殊情况）"""
        for node in self.all_nodes:
            if node.function in ["max", "min"] and not any(c.token_type == "OT" for c in node.children):
                # 添加一个等于操作符作为子节点
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
        """评估树结构的有效性（基于数据库模式）"""
        self.invalid = 0
        self.weight = 0.0
        # 1. 检查函数节点是否有必要的子节点
        for node in self.all_nodes:
            if node.function in ["avg", "sum", "max", "min"]:
                if not node.children:
                    self.invalid += 1  # 函数节点必须有子节点
                else:
                    self.weight += 0.5  # 有效函数节点加分
            # 2. 检查操作符节点是否有两个子节点
            if node.token_type == "OT" and len(node.children) != 2:
                self.invalid += 1
            # 3. 检查节点映射是否符合数据库关系
            if hasattr(node, 'mapped_elements') and node.mapped_elements:
                elem = node.mapped_elements[0].schema_element
                # 检查当前节点与父节点的映射是否符合数据库关系
                if node.parent and hasattr(node.parent, 'mapped_elements') and node.parent.mapped_elements:
                    parent_elem = node.parent.mapped_elements[0].schema_element
                    # 从schema_graph中获取与父节点元素关联的所有元素
                    related_elements = schema_graph.get_related_elements(parent_elem)
                    if elem in related_elements:
                        self.weight += 0.3  # 符合关系加分

    def hash_tree_to_number(self):
        """计算树的哈希值（用于去重）"""
        node_hashes = [f"{n.node_id}:{n.label}:{n.parent.node_id if n.parent else 'None'}" for n in self.all_nodes]
        self.hash_num = hash("|".join(sorted(node_hashes)))

    def __lt__(self, other: 'Tree') -> bool:
        """排序：无效性低的优先，成本低的优先"""
        if self.invalid != other.invalid:
            return self.invalid < other.invalid
        return (self.weight * 10000 - self.cost) > (other.weight * 10000 - other.cost)


class TreeNode:
    """用于树结构调整的节点类（扩展ParseTreeNode功能）"""
    def __init__(self, node_id: int, label: str, token_type: str, function: str, parent: Optional['TreeNode']):
        self.node_id = node_id  # 节点ID
        self.label = label  # 节点文本
        self.token_type = token_type  # 节点类型（NT/VTTEXT等）
        self.function = function  # 函数/操作符功能（avg/=/etc）
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.mapped_elements = []  # 映射的数据库元素

    def __repr__(self) -> str:
        return f"TreeNode({self.label}, type={self.token_type}, id={self.node_id})"


class TreeStructureAdjustor:
    @staticmethod
    def tree_structure_adjust(query: Query, schema_graph: SchemaGraph) -> None:
        """树结构调整主函数（接收SchemaGraph而非db）"""
        print("\n----- 步骤8: 树结构调整 (Tree Structure Adjustment) -----")
        if not query.parse_tree or not query.parse_tree.nodes:
            print("解析树为空，无法进行结构调整")
            return
        
        # 初始化调整后的树列表
        query.adjusting_trees = []
        pre_trees = {}  # 用于去重的哈希表（哈希值 -> 成本）

        # 执行调整流程（传入schema_graph）
        TreeStructureAdjustor.adjust(query, schema_graph, add_equal=False, pre_trees=pre_trees)
        
        # 检查是否需要补充调整（针对max/min等函数）
        has_max_min = any(node.function in ["max", "min"] for node in query.parse_tree.nodes if hasattr(node, 'function'))
        if (not query.adjusting_trees or (query.adjusting_trees and query.adjusting_trees[0].cost > 3)) and has_max_min:
            TreeStructureAdjustor.adjust(query, schema_graph, add_equal=True, pre_trees=pre_trees)

        # 排序并去重调整后的树
        adjusting_trees = query.adjusting_trees
        adjusting_trees.sort()
        TreeStructureAdjustor._deduplicate_trees(adjusting_trees)

        # 构建最终的调整后树（转换为ParseTree格式）
        TreeStructureAdjustor.build_adjusted_trees(query)
        print(f"树结构调整完成，保留 {len(query.adjusted_trees)} 个有效树结构")

    @staticmethod
    def pre_adjust(tree: Tree) -> None:
        """预调整：清理明显错误的节点结构"""
        # 处理函数节点（avg/sum）无参数的情况
        for node in tree.all_nodes:
            if node.function in ["avg", "sum"] and not node.children:
                if node.parent:
                    tree.move_sub_tree(node, node.parent)
                    print(f"预调整：移动无参数函数节点 {node.label} 到父节点下")
            # 处理操作符节点（OT）无参数的情况
            if node.token_type == "OT" and not node.children and node.parent:
                tree.move_sub_tree(node, node.parent)
                print(f"预调整：移动无参数操作符节点 {node.label} 到父节点下")

    @staticmethod
    def adjust(query: Query, schema_graph: SchemaGraph, add_equal: bool, pre_trees: Dict[int, int]) -> None:
        """扩展并评估树结构（使用schema_graph）"""
        # 基于原始解析树创建初始树
        initial_tree = Tree(query.parse_tree)
        TreeStructureAdjustor.pre_adjust(initial_tree)
        
        # 处理max/min等函数的特殊情况（添加等于操作符）
        if add_equal:
            initial_tree.add_equal()
        
        # 评估初始树的有效性（传入schema_graph）
        initial_tree.tree_evaluation(schema_graph, query)
        initial_tree.hash_tree_to_number()
        
        # 保存有效的初始树
        if initial_tree.invalid == 0:
            query.adjusting_trees.append(initial_tree)
        
        # 用队列扩展树结构（广度优先搜索）
        queue = [initial_tree]
        pre_trees[initial_tree.hash_num] = initial_tree.cost
        
        # 限制队列大小（避免过度扩展）
        while queue and len(queue) < 100:
            current_tree = queue.pop(0)
            
            # 扩展当前树，生成新的可能结构（传入schema_graph）
            extended_trees = TreeStructureAdjustor.extend(current_tree, schema_graph, query)
            for new_tree in extended_trees:
                new_tree.hash_tree_to_number()
                # 去重或更新更优的树（成本更低）
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
        """扩展树结构：通过移动节点生成新树"""
        extended_trees = []
        if current_tree.cost > 4:  # 限制最大成本（调整次数）
            return extended_trees
        
        # 尝试移动每个节点到其他节点下，生成新树
        for node in current_tree.all_nodes[1:]:  # 跳过根节点
            extended = TreeStructureAdjustor.extend_node(current_tree, node, schema_graph, query)
            extended_trees.extend(extended)
        return extended_trees

    @staticmethod
    def extend_node(current_tree: Tree, node: TreeNode, schema_graph: SchemaGraph, query: Query) -> List[Tree]:
        """移动单个节点生成新树（使用schema_graph评估）"""
        extended_trees = []
        # 尝试将节点移动到其他每个节点下
        for target_parent in current_tree.all_nodes:
            if target_parent.node_id == node.node_id:
                continue  # 不能移动到自身下
            # 克隆当前树
            new_tree = copy.deepcopy(current_tree)
            new_tree.cost += 1  # 增加成本
            # 找到新树中的对应节点
            new_node = next(n for n in new_tree.all_nodes if n.node_id == node.node_id)
            new_target_parent = next(n for n in new_tree.all_nodes if n.node_id == target_parent.node_id)
            # 移动节点
            if new_tree.move_sub_tree(new_node, new_target_parent):
                # 评估新树（传入schema_graph）
                new_tree.tree_evaluation(schema_graph, query)
                # 保留更优的树
                if (new_tree.invalid < current_tree.invalid or 
                    (new_tree.invalid == current_tree.invalid and 
                     new_tree.weight * 10000 - new_tree.cost > current_tree.weight * 10000 - current_tree.cost)):
                    extended_trees.append(new_tree)
        return extended_trees

    @staticmethod
    def _deduplicate_trees(trees: List[Tree]) -> None:
        """移除重复的树结构（通过哈希值）"""
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
        """将调整后的树转换为ParseTree格式"""
        query.adjusted_trees = []
        if not hasattr(query, 'adjusting_trees'):
            return  # 无调整树则直接返回
        
        adjusting_trees = query.adjusting_trees[:5]  # 保留前5个最优树
        
        for adj_tree in adjusting_trees:
            # 克隆原始解析树作为基础
            adjusted_tree = copy.deepcopy(query.parse_tree)
            # 补充缺失的节点
            node_id_map = {n.wordOrder: n for n in adjusted_tree.nodes}
            for tn in adj_tree.all_nodes:
                if tn.node_id not in node_id_map:
                    # 创建新的ParseTreeNode补充
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
            
            # 重建父节点-子节点关系
            for tn in adj_tree.all_nodes:
                parse_node = node_id_map[tn.node_id]
                # 更新父节点
                if tn.parent:
                    parse_node.parent = node_id_map.get(tn.parent.node_id)
                # 清空并重建子节点
                parse_node.children = []
                for child_tn in tn.children:
                    child_node = node_id_map.get(child_tn.node_id)
                    if child_node:
                        parse_node.children.append(child_node)
            
            # 修复根节点
            adjusted_tree.root = next((n for n in adjusted_tree.nodes if n.parent is None), None)
            query.adjusted_trees.append(adjusted_tree)
        
        # 设置默认使用的树
        if query.adjusted_trees:
            query.query_tree = query.adjusted_trees[0]
            print("已选择最优调整树作为查询树")