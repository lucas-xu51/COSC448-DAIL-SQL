# 导入必要的模块和类
from parse_tree.component.stanzaParser import StanfordNLParser
from parse_tree.component.node_mapper_init_step import NodeMapper
from parse_tree.component.query import Query  # 仅导入Query类用于封装数据
import json
import os
from tqdm import tqdm  # 导入进度条库
from pathlib import Path

def main():
    # 1. 配置文件路径（原始数据集文件）
    input_files = [
        {
            "path": "C:/Users/grizz/OneDrive/Desktop/COSC448/ideas/model/DAIL-SQL/dataset/cosc304/dev.json",
            "type": "dev"  # 标记为验证集
        },
        {
            "path": "C:/Users/grizz/OneDrive/Desktop/COSC448/ideas/model/DAIL-SQL/dataset/cosc304/train.json",
            "type": "train"  # 标记为训练集
        }
    ]
    tokens_path = "C:/Users/grizz/OneDrive/Desktop/COSC448/ideas/model/DAIL-SQL/parse_tree/zfiles/tokens.xml"
    
    # 定义输出根目录（在cosc304下创建compressed_results文件夹）
    output_root = "C:/Users/grizz/OneDrive/Desktop/COSC448/ideas/model/DAIL-SQL/dataset/cosc304/compressed_results"
    Path(output_root).mkdir(parents=True, exist_ok=True)  # 确保文件夹存在

    # 2. 初始化解析器
    print("\n===== 初始化解析器 =====")
    parser = StanfordNLParser()

    # 3. 分别处理dev和train文件
    for file_info in input_files:
        file_path = file_info["path"]
        data_type = file_info["type"]
        output_file = os.path.join(output_root, f"{data_type}_compressed_columns.jsonl")

        if not os.path.exists(file_path):
            print(f"⚠️ 警告：文件不存在 - {file_path}，已跳过该文件")
            continue
        
        # 加载当前数据集
        print(f"\n===== 开始处理 {data_type} 集 =====")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"✅ 成功加载 {file_path}，共 {len(data)} 条数据")

        # 处理并写入结果（覆盖模式）
        success_count = 0
        with open(output_file, "w", encoding="utf-8") as f_out:
            for idx, item in enumerate(tqdm(data, desc=f"处理{data_type}数据", unit="条")):
                # 实例化Query对象
                try:
                    query = Query(
                        raw_question=item["question"],
                        question_tokens=item["question_toks"],
                        schema_graph=None
                    )
                except KeyError as e:
                    tqdm.write(f"❌ 第{idx+1}条数据缺少必要字段 {e}，已跳过")
                    continue

                # 生成解析树
                try:
                    parser.parse(query)
                    if not (query.parse_tree and query.parse_tree.root):
                        tqdm.write(f"❌ 第{idx+1}条数据解析树生成失败，已跳过")
                        continue
                except Exception as e:
                    tqdm.write(f"❌ 第{idx+1}条数据解析时出错: {str(e)}，已跳过")
                    continue

                # 执行短语映射
                try:
                    compressed_pairs = NodeMapper.phrase_process(query, tokens_path)
                    if not compressed_pairs:
                        tqdm.write(f"⚠️ 第{idx+1}条数据未获取到有效压缩结果，已跳过")
                        continue
                except Exception as e:
                    tqdm.write(f"❌ 第{idx+1}条数据压缩时出错: {str(e)}，已跳过")
                    continue

                # 关键修改：确保索引从0开始（若原始索引从1开始，则减1；若已为0则不变）
                adjusted_pairs = []
                for token, idx_in_original in compressed_pairs:
                    # 强制转为0基索引（处理可能的1基索引）
                    adjusted_idx = idx_in_original - 1 if idx_in_original > 0 else idx_in_original
                    adjusted_pairs.append((token, adjusted_idx))

                # 构造并写入JSONL数据
                json_data = {
                    "compressed_column": [pair[0] for pair in adjusted_pairs],
                    "compressed_id": [pair[1] for pair in adjusted_pairs]
                }
                f_out.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                success_count += 1

        # 输出当前数据集处理统计
        print(f"\n===== {data_type} 集处理完成 =====")
        print(f"📊 总数据量: {len(data)} 条")
        print(f"✅ 成功处理: {success_count} 条")
        print(f"❌ 失败/跳过: {len(data) - success_count} 条")
        print(f"💾 结果保存至: {os.path.abspath(output_file)}")

    print("\n===== 所有数据集处理完毕 =====")
    print(f"📁 所有结果均保存在: {os.path.abspath(output_root)}")

if __name__ == "__main__":
    main()