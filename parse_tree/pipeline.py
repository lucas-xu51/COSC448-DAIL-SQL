# 导入必要的模块和类
from component.query import load_queries_from_jsonl, Query
from component.stanzaParser import StanfordNLParser
from component.node_mapper import NodeMapper
from component.entity_resolution import EntityResolution
from component.tree_structure_adjustor import TreeStructureAdjustor
from component.explainer import Explainer  # 导入Explainer组件
import json

def main():
    # 1. 配置文件路径
    jsonl_path = "C:/Users/grizz/OneDrive/Desktop/COSC448/ideas/model/DAIL-SQL/parse_tree/zfiles/test.jsonl"
    tokens_path = "C:/Users/grizz/OneDrive/Desktop/COSC448/ideas/model/DAIL-SQL/parse_tree/zfiles/tokens.xml"
    
    # 2. 从JSONL文件加载查询（生成Query对象，每个query已包含schema_graph）
    print("===== 加载查询数据 =====")
    queries = load_queries_from_jsonl(jsonl_path)
    if not queries:
        print("未加载到任何查询数据，请检查JSONL文件路径是否正确。")
        return
    
    # 3. 初始化Stanza解析器
    print("\n===== 初始化解析器 =====")
    parser = StanfordNLParser()
    
    # 4. 对每个查询生成解析树并进行处理
    for i, query in enumerate(queries, 1):
        print(f"\n===== 处理第 {i} 个查询 =====")
        print(f"原始问题: {query.sentence['raw_question']}")
        
        # 生成解析树
        parser.parse(query)
        
        # 验证解析树生成结果并打印
        if query.parse_tree and query.parse_tree.root:
            print("\n----- 生成的句法分析树 -----")
            query.parse_tree.print_tree()
        else:
            print("\n警告：解析树生成失败！")
            continue
        
        # 打印树表和并列关系表
        print("\n----- 树表内容 -----")
        for entry in query.treeTable:
            print(entry)
        
        print("\n----- 并列关系表 -----")
        print(query.conjTable if query.conjTable else "无并列关系")
        
        # 5. 执行短语映射（语义初步处理）
        print("\n----- 执行短语映射 -----")
        try:
            NodeMapper.phrase_process(query, tokens_path)
            
            # 打印映射结果
            print("\n----- 短语映射结果 -----")
            for node in query.parse_tree.nodes:
                if hasattr(node, 'mapped_elements') and node.mapped_elements:
                    print(f"节点: {node.label} ({node.token_type})")
                    print(f"映射元素: {[str(e) for e in node.mapped_elements[:3]]}")
                    print("-" * 40)
            
            # 6. 执行实体解析
            EntityResolution.entity_resolute(query)
            
            # 7. 执行树结构调整（使用query.graph作为数据库schema来源）
            print("\n----- 执行树结构调整 -----")
            TreeStructureAdjustor.tree_structure_adjust(query, query.graph)
            
            # 打印调整后的树结构（可选）
            if hasattr(query, 'adjusted_trees') and query.adjusted_trees and query.adjusted_trees[0].root:
                print("\n----- 调整后的最优句法树 -----")
                query.adjusted_trees[0].print_tree()
            
            # 8. 新增：执行自然语言解释生成
            print("\n----- 生成自然语言解释 -----")
            Explainer.explain(query)
            
            # 打印生成的自然语言解释
            if hasattr(query, 'nl_sentences') and query.nl_sentences:
                for j, nl_sentence in enumerate(query.nl_sentences, 1):
                    words = nl_sentence['words']
                    explanation = " ".join(words)
                    print(f"自然语言解释 {j}: {explanation}")
            
            # 保存所有结果到文件
            save_mapping_result(query, i)
            
        except Exception as e:
            print(f"处理过程中出错: {e}")
            continue

def save_mapping_result(query: Query, query_index: int):
    """保存映射结果、实体解析结果、树结构调整结果和自然语言解释到JSON文件"""
    result = {
        "original_question": query.sentence['raw_question'],
        "mapped_nodes": [],
        "entities": [],
        "adjusted_trees_count": len(query.adjusted_trees) if hasattr(query, 'adjusted_trees') else 0,
        "nl_explanations": []  # 新增：自然语言解释
    }
    
    # 保存映射节点
    for node in query.parse_tree.nodes:
        if hasattr(node, 'mapped_elements') and node.mapped_elements:
            node_data = {
                "label": node.label,
                "token_type": node.token_type,
                "mapped_elements": [
                    {
                        "element_name": elem.schema_element.name,
                        "element_type": elem.schema_element.type,
                        "similarity": elem.similarity,
                        "mapped_values": elem.mapped_values
                    }
                    for elem in node.mapped_elements
                ]
            }
            result["mapped_nodes"].append(node_data)
    
    # 保存实体解析结果
    if hasattr(query, 'entities') and query.entities:
        for pair in query.entities:
            entity_data = {
                "left_node": {
                    "label": pair.left_node.label,
                    "token_type": pair.left_node.token_type,
                    "word_order": pair.left_node.wordOrder
                },
                "right_node": {
                    "label": pair.right_node.label,
                    "token_type": pair.right_node.token_type,
                    "word_order": pair.right_node.wordOrder
                },
                "relation": pair.relation
            }
            result["entities"].append(entity_data)
    
    # 保存自然语言解释
    if hasattr(query, 'nl_sentences') and query.nl_sentences:
        for nl_sentence in query.nl_sentences:
            result["nl_explanations"].append({
                "words": nl_sentence['words'],
                "explanation": " ".join(nl_sentence['words'])
            })
    
    # 保存到文件
    output_path = f"C:/Users/grizz/OneDrive/Desktop/COSC448/ideas/model/DAIL-SQL/parse_tree/zfiles/query_{query_index}_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    main()