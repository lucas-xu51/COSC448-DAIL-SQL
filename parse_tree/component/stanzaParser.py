import stanza
from typing import List, Dict, Optional, Tuple
from .query import Query  # 从query.py导入Query类

class ParseTreeNode:
    """句法分析树的节点"""
    def __init__(self, label: str, word_order: int, pos: str, 
                 relationship: str, parent: Optional['ParseTreeNode'] = None):
        self.label = label           # 节点对应的词文本
        self.wordOrder = word_order  # 词在句子中的位置（从1开始）
        self.pos = pos               # 词性标签
        self.relationship = relationship  # 与父节点的依存关系
        self.parent = parent         # 父节点引用
        self.children = []           # 子节点列表
        self.leftRel = None          # 并列关系标记

    def __repr__(self) -> str:
        return f"Node({self.label}, pos={self.pos}, rel={self.relationship})"

class ParseTree:
    """句法分析树"""
    def __init__(self):
        self.root = None  # 根节点
        self.nodes = []   # 所有节点的列表

    def build_node(self, node_info: Tuple[str, str, str, str, str]) -> bool:
        """根据树表条目构建节点并添加到树中"""
        dep_index, dep_value, pos, gov_index, relationship = node_info
        dep_index = int(dep_index)
        gov_index = int(gov_index)
        
        # 创建新节点
        new_node = ParseTreeNode(
            label=dep_value,
            word_order=dep_index,
            pos=pos,
            relationship=relationship
        )
        
        # 根节点处理
        if gov_index == 0:
            self.root = new_node
            self.nodes.append(new_node)
            return True
        
        # 查找父节点
        parent_node = self.search_node_by_order(gov_index)
        if parent_node:
            new_node.parent = parent_node
            parent_node.children.append(new_node)
            self.nodes.append(new_node)
            return True
        
        return False  # 父节点未找到，稍后再试

    def search_node_by_order(self, word_order: int) -> Optional[ParseTreeNode]:
        """根据词序查找节点"""
        for node in self.nodes:
            if node.wordOrder == word_order:
                return node
        return None

    def print_tree(self, node: Optional[ParseTreeNode] = None, level: int = 0):
        """递归打印树结构（用于调试）"""
        if node is None:
            node = self.root
        
        indent = "  " * level
        rel_info = f", rel={node.leftRel}" if node.leftRel else ""
        print(f"{indent}{node} (order={node.wordOrder}{rel_info})")
        
        for child in node.children:
            self.print_tree(child, level + 1)

    def delete_node(self, node: ParseTreeNode) -> None:
        """从树中删除指定节点，并将其子节点连接到其父节点"""
        if node not in self.nodes:
            return  # 节点不在树中，直接返回
        
        # 1. 将子节点连接到父节点
        if node.parent and node.children:
            for child in node.children:
                child.parent = node.parent
                if child not in node.parent.children:
                    node.parent.children.append(child)
        
        # 2. 从父节点的子节点列表中移除
        if node.parent and node in node.parent.children:
            node.parent.children.remove(node)
        
        # 3. 从节点列表中移除
        self.nodes.remove(node)
        
        # 4. 特殊处理：如果删除的是根节点，尝试寻找合适的新根
        if node == self.root:
            # 选择第一个没有父节点的节点作为新根
            self.root = next((n for n in self.nodes if n.parent is None), None)
            if not self.root and self.nodes:
                # 如果没有这样的节点，将第一个节点设为根，并将其父节点设为None
                self.root = self.nodes[0]
                self.root.parent = None

class StanfordNLParser:
    """自然语言解析器，生成句法分析树"""
    def __init__(self):
        # 初始化stanza解析器
        self.nlp = stanza.Pipeline(
            lang='en', 
            processors='tokenize,pos,lemma,depparse',
            tokenize_pretokenized=True  # 假设输入已经分词
        )

    def parse(self, query: Query) -> None:
        """解析查询并构建句法分析树"""
        self._stanford_parse(query)
        self._build_tree(query)
        self._fix_conj(query)

    def _stanford_parse(self, query: Query) -> None:
        """使用Stanford Parser进行句法分析"""
        # 将分词结果转换为stanza可以处理的格式
        doc = self.nlp([query.sentence["question_tokens"]])
        
        # 处理依存句法分析结果
        tree_table = []  # 存储树表条目
        conj_table = []  # 存储并列关系
        
        # 假设只有一个句子
        sentence = doc.sentences[0]
        
        # 构建树表
        for dep in sentence.dependencies:
            # 依存关系格式: (governor, relation, dependent)
            governor = dep[0]
            relation = dep[1]
            dependent = dep[2]
            
            # 注意：stanza的索引从1开始，0表示根
            dep_index = dependent.id
            dep_value = dependent.text
            pos = dependent.xpos  # 使用XPOS (Penn Treebank标签)
            gov_index = governor.id
            
            # 构建树表条目
            tree_table_entry = (str(dep_index), dep_value, pos, str(gov_index), relation)
            tree_table.append(tree_table_entry)
            
            # 处理并列关系
            if relation.startswith('conj'):
                conj_entry = f"{gov_index} {dep_index}"
                conj_table.append(conj_entry)
        
        # 将结果存入query对象
        query.treeTable = tree_table
        query.conjTable = conj_table

    def _build_tree(self, query: Query) -> None:
        """根据树表构建句法分析树"""
        # 修改：使用query.parse_tree而不是query.parseTree
        query.parse_tree = ParseTree()
        
        # 标记已处理的条目
        done_list = [False] * len(query.treeTable)
        
        # 首先处理根节点
        for i, entry in enumerate(query.treeTable):
            if entry[3] == "0":  # 父节点是根(0)
                query.parse_tree.build_node(entry)
                done_list[i] = True
        
        # 循环处理剩余节点，直到所有节点都被处理
        while not all(done_list):
            progress = False
            for i, entry in enumerate(query.treeTable):
                if not done_list[i]:
                    if query.parse_tree.build_node(entry):
                        done_list[i] = True
                        progress = True
                        break
            
            # 如果某次循环没有处理任何节点，说明存在问题
            if not progress:
                break

    def _fix_conj(self, query: Query) -> None:
        """修复并列关系，设置leftRel属性"""
        if not query.conjTable:
            return
        
        for conj in query.conjTable:
            gov_num, dep_num = map(int, conj.split())
            gov_node = query.parse_tree.search_node_by_order(gov_num)
            dep_node = query.parse_tree.search_node_by_order(dep_num)
            
            if not gov_node or not dep_node:
                continue
            
            # 确定并列关系逻辑词
            logic = ","
            prev_node = query.parse_tree.search_node_by_order(dep_node.wordOrder - 1)
            if prev_node:
                logic = prev_node.label.lower()
            
            # 设置并列关系标记
            if logic == "or":
                dep_node.leftRel = "or"
                # 检查 gov_node.parent 是否存在，避免 None 错误
                if gov_node.parent:
                    for sibling in gov_node.parent.children:
                        if sibling.leftRel == ",":
                            sibling.leftRel = "or"
            elif logic in ("and", "but"):
                dep_node.leftRel = "and"
                # 检查 gov_node.parent 是否存在
                if gov_node.parent:
                    for sibling in gov_node.parent.children:
                        if sibling.leftRel == ",":
                            sibling.leftRel = "and"
            else:
                dep_node.leftRel = ","
            
            # 调整树结构：将 dep_node 移到与 gov_node 同级
            # 先检查 dep_node 是否在 gov_node 的子节点中
            if dep_node in gov_node.children:
                gov_node.children.remove(dep_node)
            
            # 关键修复：确保父节点存在后再操作
            if gov_node.parent is not None:
                dep_node.parent = gov_node.parent
                # 检查 dep_node 是否已在父节点的子列表中，避免重复添加
                if dep_node not in dep_node.parent.children:
                    dep_node.parent.children.append(dep_node)
            else:
                # 若 gov_node 是根节点（无父节点），直接将 dep_node 设为根节点的子节点
                dep_node.parent = gov_node
                if dep_node not in dep_node.parent.children:
                    dep_node.parent.children.append(dep_node)
            
            # 继承关系类型
            dep_node.relationship = gov_node.relationship

# 示例使用
if __name__ == "__main__":
    # 假设我们已经有一个Query对象
    # 这里简单创建一个示例
    from query import SchemaGraph  # 假设SchemaGraph在另一个文件中
    
    # 加载数据库模式信息（示例中使用空数据）
    db_info = {
        "tables": [["stadium"], ["singer"], ["concert"], ["singer", "in", "concert"]],
        "columns": [...]  # 省略完整列定义
    }
    schema_graph = SchemaGraph(db_info)
    
    # 创建查询对象
    query = Query(
        raw_question="How many singers do we have?",
        question_tokens=["how", "many", "singer", "do", "we", "have", "?"],
        schema_graph=schema_graph
    )
    
    # 初始化解析器并解析查询
    parser = StanfordNLParser()
    parser.parse(query)
    
    # 打印解析树（调试用）
    # 修改：使用query.parse_tree而不是query.parseTree
    if query.parse_tree:
        print("句法分析树结构:")
        query.parse_tree.print_tree()
    
    # 打印树表（调试用）
    print("\n树表内容:")
    for entry in query.treeTable:
        print(entry)