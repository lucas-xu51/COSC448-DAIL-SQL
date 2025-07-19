from typing import List, Dict, Optional, Tuple
from .query import Query, SchemaGraph
from .stanzaParser import ParseTree, ParseTreeNode
from .tree_structure_adjustor import Tree, TreeNode
import re

class Explainer:
    @staticmethod
    def explain(query: Query) -> None:
        """为查询的每个调整树生成自然语言解释"""
        if not hasattr(query, 'adjusted_trees') or not query.adjusted_trees:
            print("没有调整后的树可供解释")
            return
        
        query.nl_sentences = []  # 存储生成的自然语言句子
        for i, tree in enumerate(query.adjusted_trees):
            try:
                nl = Explainer.explain_tree(tree)
                if nl:
                    query.nl_sentences.append(nl)
            except Exception as e:
                print(f"生成解释时出错: {e}")
    
    @staticmethod
    def explain_tree(tree: ParseTree) -> Optional[Dict]:
        """将解析树转换为自然语言句子，更通用地处理不同树结构"""
        nl = {
            'words': [],       # 生成的自然语言单词列表
            'nodes': [],       # 对应的解析树节点
            'is_implicit': []  # 标记每个单词是否为隐式添加
        }
        
        # 尝试找到根节点(支持"ROOT"标签或实际根节点)
        root = None
        if tree.nodes and tree.nodes[0].label == "ROOT":
            root = tree.nodes[0]
        else:
            # 尝试找到实际的根节点(关系为root的节点)
            for node in tree.nodes:
                if hasattr(node, 'rel') and node.rel == 'root':
                    root = node
                    break
        
        if not root:
            print("警告：未找到根节点")
            return None
        
        # 处理根节点(如果是ROOT类型，使用其第一个子节点作为操作)
        if root.label == "ROOT":
            if not root.children:
                return None
            root_op = root.children[0]
            Explainer._add_node(nl, root_op, root_op.label, False)
        else:
            # 直接使用识别出的根节点
            Explainer._add_node(nl, root, root.label, False)
        
        # 查找核心名词短语(NT类型)
        core_nt = None
        if root.label == "ROOT" and root.children:
            # 从ROOT的子节点中查找NT
            for child in root.children:
                if child.token_type == "NT":
                    core_nt = child
                    break
        else:
            # 从实际根节点的子节点中查找
            for child in root.children:
                if child.token_type == "NT":
                    core_nt = child
                    break
        
        if not core_nt:
            # 尝试更宽松的查找
            for node in tree.nodes:
                if node.token_type == "NT":
                    core_nt = node
                    break
        
        if not core_nt:
            print("警告：未找到核心名词短语")
            return None
        
        # 处理核心名词短语
        add_the = True  # 默认添加"the"
        Explainer._add_core_nt(core_nt, add_the, nl)
        
        # 处理条件子句(WHERE部分)
        has_where = False
        for node in tree.nodes:
            # 查找操作符节点(OT类型)
            if node.token_type != "OT" or len(getattr(node, 'children', [])) != 2:
                continue
            
            # 添加"where"引导词
            if not has_where:
                Explainer._add_node(nl, None, "where", False)
                has_where = True
            
            left, right = node.children
            
            # 处理左子节点
            if left.token_type == "FT":
                Explainer._add_node(nl, left, left.label, False)
            
            # 处理核心名词短语
            if left.token_type == "NT":
                Explainer._add_core_nt(left, True, nl)
            
            # 添加操作符
            if node.function != "=":
                ot_text = Explainer._to_ot(node)
                Explainer._add_node(nl, node, f"is {ot_text}", False)
            else:
                Explainer._add_node(nl, node, "is", False)
            
            # 处理右子节点
            if right.token_type == "NT":
                Explainer._add_core_nt(right, True, nl)
            elif right.token_type == "VTTEXT":
                Explainer._add_node(nl, right, right.label, False)
        
        return nl
    
    @staticmethod
    def _add_core_nt(core_nt: ParseTreeNode, add_the: bool, nl: Dict) -> None:
        """递归添加核心名词短语及其子节点，更通用地处理不同结构"""
        node_stack = [core_nt]
        
        while node_stack:
            current = node_stack.pop()
            label = ""
            
            # 添加"the"前缀
            if current == core_nt and add_the:
                label += "the "
            
            # 处理介词
            if hasattr(current, 'prep') and current.prep:
                label += f"{current.prep} "
            
            # 添加量词
            if hasattr(current, 'QT') and current.QT:
                label += f"{current.QT} "
            
            # 添加节点标签
            if len(current.label.split()) > 1:
                label += f"\"{current.label}\""
            else:
                label += current.label
            
            # 添加到自然语言结果
            is_added = hasattr(current, 'is_added') and current.is_added
            Explainer._add_node(nl, current, label, is_added)
            
            # 将子节点压入栈(逆序以保持正确顺序)
            for child in reversed(getattr(current, 'children', [])):
                node_stack.append(child)
    
    @staticmethod
    def _add_node(nl: Dict, node: Optional[ParseTreeNode], word: str, is_implicit: bool) -> None:
        """添加节点及其对应的自然语言单词"""
        nl['words'].append(word)
        nl['nodes'].append(node)
        nl['is_implicit'].append(is_implicit)
    
    @staticmethod
    def _sub_tree_contains(node: ParseTreeNode, target: ParseTreeNode) -> bool:
        """检查子树是否包含目标节点"""
        if node == target:
            return True
        for child in getattr(node, 'children', []):
            if Explainer._sub_tree_contains(child, target):
                return True
        return False
    
    @staticmethod
    def _lemmatize(word: str) -> str:
        """简单的词形还原"""
        if word.endswith('s'):
            return word[:-1]
        return word
    
    @staticmethod
    def _to_ot(node: ParseTreeNode) -> str:
        """将操作符节点转换为自然语言表示"""
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