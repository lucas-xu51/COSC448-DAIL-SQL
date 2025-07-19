import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from .query import Query, SchemaGraph, SchemaElement
from .stanzaParser import ParseTree, ParseTreeNode
import string
import math
from collections import deque

# 相似度计算函数
class SimFunctions:
    @staticmethod
    def similarity(node: ParseTreeNode, mapped_element: 'MappedSchemaElement') -> float:
        """计算节点与映射元素的相似度"""
        # 字符相似度
        char_sim = SimFunctions.char_similarity(node.label, mapped_element.schema_element.name)
        
        # 词性匹配度
        pos_sim = SimFunctions.pos_similarity(node.pos, mapped_element.schema_element)
        
        # 综合相似度
        total_sim = (char_sim * 0.6) + (pos_sim * 0.4)
        mapped_element.similarity = total_sim
        return total_sim
    
    @staticmethod
    def char_similarity(s1: str, s2: str) -> float:
        """计算字符串相似度（简化版）"""
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        
        if s1 == s2:
            return 1.0
            
        # 计算最长公共子序列长度
        lcs_length = SimFunctions.lcs(s1, s2)
        return lcs_length / max(len(s1), len(s2))
    
    @staticmethod
    def lcs(s1: str, s2: str) -> int:
        """最长公共子序列长度"""
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
        """词性匹配度"""
        # 如果是表，名词性词性更匹配
        if schema_element.type == "table" and pos.startswith("NN"):
            return 1.0
        # 如果是字段，名词性词性更匹配
        if schema_element.type == "column" and pos.startswith("NN"):
            return 0.9
        return 0.5
    
    @staticmethod
    def lemmatize(word: str) -> str:
        """简单词形还原（实际应用中可使用NLTK等库）"""
        # 简化版实现
        if word.endswith("s") and len(word) > 1:
            return word[:-1]
        return word

# 映射的模式元素类
class MappedSchemaElement:
    def __init__(self, schema_element: SchemaElement):
        self.schema_element = schema_element  # 映射的模式元素
        self.similarity = 0.0  # 相似度
        self.mapped_values = []  # 映射的值
        self.choice = 0  # 选择标记
        self.score = 0.0  # 评分

    def __repr__(self) -> str:
        return f"MappedSchemaElement({self.schema_element.name}, similarity={self.similarity:.2f})"

    def __lt__(self, other: 'MappedSchemaElement') -> bool:
        # 用于排序，相似度高的在前
        return self.similarity > other.similarity

class NodeMapper:
    @staticmethod
    def phrase_process(query: Query, tokens_path: str) -> None:
        """短语映射主流程"""
        print("\n===== 开始短语映射流程 =====")
        # 解析tokens.xml文件
        try:
            tokens_tree = ET.parse(tokens_path)
            tokens_root = tokens_tree.getroot()
            print(f"成功加载tokens文件: {tokens_path}")
        except Exception as e:
            print(f"加载tokens文件失败: {e}")
            return
        
        # 执行各步骤
        NodeMapper.tokenize(query, tokens_root)
        NodeMapper.delete_useless(query)
        NodeMapper.map_nodes(query)
        NodeMapper.delete_no_match(query)

        print("\n----- 优化后的解析树结构（排序前） -----")
        if query.parse_tree and query.parse_tree.root:
            query.parse_tree.print_tree()  # 调用ParseTree的打印方法
        else:
            print("解析树为空或根节点缺失")

        NodeMapper.individual_ranking(query)
        NodeMapper.group_ranking(query)
        print("===== 短语映射流程完成 =====")
    
    @staticmethod
    def tokenize(query: Query, tokens_root: ET.Element) -> None:
        """节点类型标注"""
        print("\n----- 步骤1: 节点类型标注 (tokenize) -----")
        parse_tree = query.parse_tree
        if not parse_tree or not parse_tree.root:
            print("解析树为空，无法进行标注")
            return
            
        # 标记根节点
        parse_tree.root.token_type = "ROOT"
        print(f"根节点标记: {parse_tree.root.label} (类型: {parse_tree.root.token_type})")
        
        # 标记核心动词(CMT)
        cmt_count = 0
        for child in parse_tree.root.children:
            if NodeMapper.is_of_type(tokens_root, parse_tree, child, "CMT", None):
                child.token_type = "CMT"
                cmt_count += 1
                print(f"核心动词标记: {child.label} (类型: {child.token_type})")
        print(f"共标记 {cmt_count} 个核心动词(CMT)")
        
        # 标记否定词(NEG)
        neg_count = 0
        for node in parse_tree.nodes:
            if not hasattr(node, 'token_type') or node.token_type == "NA":
                node.token_type = "NA"  # 初始化未识别类型
                
            if node.token_type == "NA" and NodeMapper.is_of_type(tokens_root, parse_tree, node, "NEG", None):
                node.token_type = "NEG"
                neg_count += 1
                print(f"否定词标记: {node.label} (类型: {node.token_type})")
        print(f"共标记 {neg_count} 个否定词(NEG)")
        
        # 合并多词短语
        original_node_count = len(parse_tree.nodes)
        NodeMapper.merge_multi_word_expressions(parse_tree)
        merged_count = original_node_count - len(parse_tree.nodes)
        print(f"合并多词短语: 共合并 {merged_count} 个节点")
        
        # 其他类型标注
        current_size = 0
        type_counts = {"FT":0, "OT":0, "QT":0, "VT":0, "NTVT":0, "JJ":0, "NA":0}
        
        while current_size != len(parse_tree.nodes):
            current_size = len(parse_tree.nodes)
            for node in parse_tree.nodes[:]:  # 使用副本避免修改时出错
                if node.token_type == "NA":
                    # 函数节点(FT)
                    if NodeMapper.is_of_type(tokens_root, parse_tree, node, "FT", "function"):
                        node.token_type = "FT"
                        type_counts["FT"] += 1
                    # 操作符节点(OT)
                    elif NodeMapper.is_of_type(tokens_root, parse_tree, node, "OT", "operator"):
                        node.token_type = "OT"
                        type_counts["OT"] += 1
                    # 数量词节点(QT)
                    elif NodeMapper.is_of_type(tokens_root, parse_tree, node, "QT", "quantity"):
                        node.token_type = "QT"
                        type_counts["QT"] += 1
                    # 数值节点(VT)
                    elif NodeMapper.is_numeric(node.label):
                        node.token_type = "VT"
                        type_counts["VT"] += 1
                    # 名词性节点(NTVT)
                    elif node.pos.startswith("NN") or node.pos == "CD":
                        node.token_type = "NTVT"
                        type_counts["NTVT"] += 1
                    # 形容词节点(JJ)
                    elif node.pos.startswith("JJ"):
                        node.token_type = "JJ"
                        type_counts["JJ"] += 1
                    else:
                        type_counts["NA"] += 1
        
        # 打印类型统计
        print("\n节点类型统计:")
        for type_name, count in type_counts.items():
            if count > 0:
                print(f"  {type_name}: {count} 个节点")
        
        # 打印所有节点类型
        print("\n所有节点类型标注结果:")
        for node in parse_tree.nodes:  # 只打印前10个避免过长
            print(f"  {node.label} (POS: {node.pos}, 类型: {node.token_type})")
        # if len(parse_tree.nodes) > 10:
        #     print(f"  ... 共 {len(parse_tree.nodes)} 个节点")
    
    @staticmethod
    def merge_multi_word_expressions(parse_tree: ParseTree) -> None:
        """合并多词短语"""
        for node in parse_tree.nodes[:]:  # 使用副本避免修改时出错
            if hasattr(node, 'relationship') and node.relationship == "mwe" and node.token_type == "NA":
                original_parent_label = node.parent.label
                if node.wordOrder > node.parent.wordOrder:
                    node.parent.label += " " + node.label
                else:
                    node.parent.label = node.label + " " + node.parent.label
                print(f"合并短语: {node.label} -> 父节点变为: {node.parent.label}")
                parse_tree.delete_node(node)
    
    @staticmethod
    def delete_useless(query: Query) -> None:
        """清理无关节点"""
        print("\n----- 步骤2: 清理无关节点 (deleteUseless) -----")
        parse_tree = query.parse_tree
        if not parse_tree:
            print("解析树为空，无法清理")
            return
            
        original_node_count = len(parse_tree.nodes)
        print(f"清理前节点数量: {original_node_count}")
        
        # 保存原始解析树
        query.original_parse_tree = ParseTree()  # 简化处理
        query.original_parse_tree.root = parse_tree.root
        
        # 清理NA和QT类型节点
        deleted_nodes = []
        prepositions = []
        for node in parse_tree.nodes[:]:  # 使用副本避免修改时出错
            if hasattr(node, 'token_type') and (node.token_type == "NA" or node.token_type == "QT"):
                # 特殊处理介词
                if node.label.lower() in ["on", "in", "of", "by", "for", "with"]:
                    prep_info = f"介词 {node.label} 信息已保存"
                    if node.children:
                        node.children[0].prep = node.label
                        prep_info += f" 到子节点 {node.children[0].label}"
                    elif node.parent:
                        node.parent.prep = node.label
                        prep_info += f" 到父节点 {node.parent.label}"
                    prepositions.append(prep_info)
                
                deleted_nodes.append(node.label)
                parse_tree.delete_node(node)
        
        # 打印清理结果
        print(f"清理后节点数量: {len(parse_tree.nodes)}")
        print(f"共删除 {original_node_count - len(parse_tree.nodes)} 个节点")
        print(f"删除的节点: {deleted_nodes[:5]} {'...' if len(deleted_nodes) > 5 else ''}")
        if prepositions:
            print("处理的介词:")
            for prep in prepositions[:3]:
                print(f"  {prep}")
    
    @staticmethod
    def map_nodes(query: Query) -> None:
        """节点与数据库元素映射"""
        print("\n----- 步骤3: 节点与数据库元素映射 (map) -----")
        parse_tree = query.parse_tree
        if not parse_tree:
            print("解析树为空，无法映射")
            return
            
        # 初始化mapped_elements属性
        for node in parse_tree.nodes:
            node.mapped_elements = []
        
        # 名词性节点和形容词节点映射
        mapped_count = 0
        for node in parse_tree.nodes:
            if node.token_type in ["NTVT", "JJ"]:
                # 检查是否存在于schema中
                NodeMapper.map_schema_element(node, query.graph)
                # 如果有映射
                if node.mapped_elements:
                    mapped_count += 1
                    top_match = node.mapped_elements[0]
                    print(f"节点 {node.label} 映射到: {top_match.schema_element.name} (相似度: {top_match.similarity:.2f})")
                else:
                    node.token_type = "NA"
                    print(f"节点 {node.label} 无匹配的数据库元素，标记为NA")
            
            # 数值节点映射
            elif node.token_type == "VT":
                # 确定操作符
                ot = "="
                if hasattr(node.parent, 'token_type') and node.parent.token_type == "OT":
                    ot = node.parent.label
                elif node.children:
                    for child in node.children:
                        if hasattr(child, 'token_type') and child.token_type == "OT":
                            ot = child.label
                            if ot == "NA" and child.label.lower() == "at least":
                                ot = ">="
                
                # 映射数值
                NodeMapper.map_numeric_element(node, ot)
                node.token_type = "VTNUM"
                print(f"数值节点 {node.label} 映射 (操作符: {ot})")
                mapped_count += 1
        
        print(f"成功映射的节点数量: {mapped_count}")
    
    @staticmethod
    def map_schema_element(node: ParseTreeNode, schema_graph: SchemaGraph) -> None:
        """将节点映射到数据库模式元素"""
        # 检查与表和字段的匹配
        for elem in schema_graph.schema_elements:
            # 计算相似度
            mapped_elem = MappedSchemaElement(elem)
            SimFunctions.similarity(node, mapped_elem)
            
            # 相似度足够高才保留
            if mapped_elem.similarity > 0.3:  # 阈值可调整
                node.mapped_elements.append(mapped_elem)
        
        # 排序映射结果
        node.mapped_elements.sort()
    
    @staticmethod
    def map_numeric_element(node: ParseTreeNode, operator: str) -> None:
        """映射数值元素"""
        # 创建数值映射
        mapped_elem = MappedSchemaElement(SchemaElement(-1, f"VALUE:{node.label}", "value"))
        mapped_elem.operator = operator
        mapped_elem.mapped_values = [node.label]
        mapped_elem.similarity = 1.0  # 数值匹配度为1
        node.mapped_elements.append(mapped_elem)
    
    @staticmethod
    def delete_no_match(query: Query) -> None:
        """删除无映射节点"""
        print("\n----- 步骤4: 删除无映射节点 (deleteNoMatch) -----")
        parse_tree = query.parse_tree
        if not parse_tree:
            print("解析树为空，无法删除")
            return
            
        original_node_count = len(parse_tree.nodes)
        print(f"删除前节点数量: {original_node_count}")
        
        deleted_nodes = []
        for node in parse_tree.nodes[:]:  # 使用副本避免修改时出错
            if hasattr(node, 'token_type') and node.token_type == "NA":
                # 特殊处理介词
                if node.label.lower() in ["on", "in"] and node.parent:
                    node.parent.prep = node.label
                    print(f"介词 {node.label} 信息已保存到父节点 {node.parent.label}")
                
                deleted_nodes.append(node.label)
                parse_tree.delete_node(node)
        
        print(f"删除后节点数量: {len(parse_tree.nodes)}")
        print(f"共删除 {original_node_count - len(parse_tree.nodes)} 个无映射节点")
        print(f"删除的节点: {deleted_nodes}")
    
    @staticmethod
    def individual_ranking(query: Query) -> None:
        """单节点映射排序"""
        print("\n----- 步骤5: 单节点映射排序 (individualRanking) -----")
        parse_tree = query.parse_tree
        if not parse_tree:
            print("解析树为空，无法排序")
            return
            
        ranked_nodes = 0
        for node in parse_tree.nodes:
            if hasattr(node, 'mapped_elements') and node.mapped_elements:
                # 确保已计算相似度
                for mapped_elem in node.mapped_elements:
                    SimFunctions.similarity(node, mapped_elem)
                
                # 排序
                node.mapped_elements.sort()
                ranked_nodes += 1
                
                # 打印排序结果
                print(f"\n节点 {node.label} 的映射排序:")
                for i, elem in enumerate(node.mapped_elements[:3]):  # 只打印前3个
                    print(f"  第{i+1}名: {elem.schema_element.name} (相似度: {elem.similarity:.4f})")
                if len(node.mapped_elements) > 3:
                    print(f"  ... 共 {len(node.mapped_elements)} 个映射")
                
                # 处理重复映射
                if node.token_type == "NTVT":
                    original_count = len(node.mapped_elements)
                    NodeMapper.handle_duplicate_mappings(node)
                    if len(node.mapped_elements) < original_count:
                        print(f"  去重后保留 {len(node.mapped_elements)} 个映射")
        
        print(f"完成 {ranked_nodes} 个节点的映射排序")
    
    @staticmethod
    def handle_duplicate_mappings(node: ParseTreeNode) -> None:
        """处理重复映射"""
        delete_list = []
        mapped_elements = node.mapped_elements
        
        for j in range(len(mapped_elements)):
            nt = mapped_elements[j]
            for k in range(j + 1, len(mapped_elements)):
                vt = mapped_elements[k]
                # 如果是同一模式元素的不同映射
                if (not nt.mapped_values and vt.mapped_values and 
                    nt.schema_element.element_id == vt.schema_element.element_id):
                    if nt.similarity >= vt.similarity:
                        vt.similarity = nt.similarity
                        vt.choice = -1
                        # 交换位置
                        mapped_elements[j], mapped_elements[k] = mapped_elements[k], mapped_elements[j]
                    delete_list.append(nt)
        
        # 移除重复项
        for elem in delete_list:
            if elem in mapped_elements:
                mapped_elements.remove(elem)
    
    @staticmethod
    def group_ranking(query: Query) -> None:
        """全局映射优化"""
        print("\n----- 步骤6: 全局映射优化 (groupRanking) -----")
        parse_tree = query.parse_tree
        if not parse_tree or not parse_tree.nodes:
            print("解析树为空，无法优化")
            return
            
        # 找到最佳根节点（映射最明确的节点）
        root = parse_tree.nodes[0]
        root_score = 0
        
        for node in parse_tree.nodes:
            if hasattr(node, 'mapped_elements') and node.mapped_elements:
                # 计算映射明确度分数
                if len(node.mapped_elements) == 1:
                    score = 1.0
                else:
                    # 分数越高表示第一名比第二名优势越明显
                    try:
                        score = 1 - node.mapped_elements[1].similarity / node.mapped_elements[0].similarity
                    except (IndexError, ZeroDivisionError):
                        score = 0
                
                if score > root_score:
                    root = node
                    root_score = score
        
        print(f"最佳根节点: {root.label} (明确度分数: {root_score:.4f})")
        if root.label == "ROOT":
            print("根节点为ROOT，无需进一步优化")
            return
            
        # 选择最佳映射
        root.choice = 0
        print(f"根节点 {root.label} 选择的最佳映射: {root.mapped_elements[0].schema_element.name}")
        
        # BFS遍历进行全局优化
        done = {node: False for node in parse_tree.nodes}
        queue = deque()
        queue.append((root, root))  # (parent, child)
        optimized_count = 0
        
        while queue:
            parent, child = queue.popleft()
            
            if not done.get(child, True):
                if parent != child and hasattr(child, 'mapped_elements') and child.mapped_elements:
                    # 计算最佳映射
                    max_score = -1
                    max_position = 0
                    
                    for i, child_elem in enumerate(child.mapped_elements):
                        if not parent.mapped_elements or parent.choice >= len(parent.mapped_elements):
                            continue
                            
                        parent_elem = parent.mapped_elements[parent.choice]
                        # 计算距离
                        try:
                            distance = parent.graph.shortest_distance[
                                parent_elem.schema_element.element_id][
                                child_elem.schema_element.element_id
                            ]
                        except:
                            distance = 1.0
                        
                        # 计算分数
                        score = parent_elem.similarity * child_elem.similarity * distance
                        
                        if score > max_score:
                            max_score = score
                            max_position = i
                    
                    child.choice = max_position
                    optimized_count += 1
                    print(f"节点 {child.label} 最佳映射: {child.mapped_elements[max_position].schema_element.name} (分数: {max_score:.4f})")
                
                # 将子节点和父节点加入队列
                for grandchild in child.children:
                    queue.append((child, grandchild))
                
                if child.parent and child.parent != child:
                    queue.append((child, child.parent))
                
                done[child] = True
        
        # 更新节点类型
        nt_count = 0
        vttext_count = 0
        for node in parse_tree.nodes:
            if node.token_type in ["NTVT", "JJ"] and hasattr(node, 'mapped_elements') and node.mapped_elements:
                if (node.choice < len(node.mapped_elements) and 
                    (not node.mapped_elements[node.choice].mapped_values or 
                     node.mapped_elements[node.choice].choice == -1)):
                    node.token_type = "NT"  # 实体节点
                    nt_count += 1
                else:
                    node.token_type = "VTTEXT"  # 值节点
                    vttext_count += 1
        
        print(f"优化完成，共处理 {optimized_count} 个节点")
        print(f"实体节点(NT)数量: {nt_count}, 值节点(VTTEXT)数量: {vttext_count}")
    
    @staticmethod
    def is_of_type(tokens_root: ET.Element, parse_tree: ParseTree, node: ParseTreeNode, token_type: str, tag: Optional[str]) -> bool:
        """判断节点是否属于特定类型"""
        # 检查两种匹配模式
        if NodeMapper._is_of_type(tokens_root, parse_tree, node, token_type, 1, tag):
            return True
        if NodeMapper._is_of_type(tokens_root, parse_tree, node, token_type, 2, tag):
            return True
        return False
    
    @staticmethod
    def _is_of_type(tokens_root: ET.Element, parse_tree: ParseTree, node: ParseTreeNode, 
                   token_type: str, case_type: int, tag: Optional[str]) -> bool:
        """判断节点是否属于特定类型的内部实现"""
        # 获取标签文本
        if case_type == 1:
            label = node.label.lower()
        else:
            label = SimFunctions.lemmatize(node.label).lower()
        
        # 在tokens.xml中查找对应类型
        token_elements = tokens_root.findall(f".//{token_type}")
        if not token_elements:
            return False
            
        for token_elem in token_elements:
            phrase_elements = token_elem.findall("phrase")
            for phrase_elem in phrase_elements:
                phrase_text = phrase_elem.text.strip().lower() if phrase_elem.text else ""
                
                # 单词语匹配
                if len(phrase_text.split()) == 1 and " " not in label:
                    if label == phrase_text:
                        if tag and phrase_elem.find(tag) is not None:
                            node.function = phrase_elem.find(tag).text.strip()
                        return True
                
                # 短语包含匹配
                elif phrase_text in label:
                    if tag and phrase_elem.find(tag) is not None:
                        node.function = phrase_elem.find(tag).text.strip()
                    return True
                    
        return False
    
    @staticmethod
    def is_numeric(s: str) -> bool:
        """判断字符串是否为数值"""
        s = s.strip()
        if not s:
            return False
            
        # 检查是否为整数或浮点数
        if s.replace('.', '', 1).isdigit():
            return True
            
        # 检查是否为带符号的数值
        if (s[0] in '+-' and len(s) > 1 and 
            s[1:].replace('.', '', 1).isdigit()):
            return True
            
        return False

# 为ParseTreeNode添加所需的属性和方法
def enhance_parse_tree_node():
    """增强ParseTreeNode类，添加所需属性"""
    # 添加token_type属性
    if not hasattr(ParseTreeNode, 'token_type'):
        ParseTreeNode.token_type = "NA"
    
    # 添加mapped_elements属性
    if not hasattr(ParseTreeNode, 'mapped_elements'):
        ParseTreeNode.mapped_elements = []
    
    # 添加prep属性
    if not hasattr(ParseTreeNode, 'prep'):
        ParseTreeNode.prep = None
    
    # 添加choice属性
    if not hasattr(ParseTreeNode, 'choice'):
        ParseTreeNode.choice = 0

# 增强ParseTreeNode类
enhance_parse_tree_node()