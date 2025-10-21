import json

# 输入文件路径
input_path = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\dataset\spider\dev_all.json"
# 输出文件路径
output_path = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\dataset\spider\cre_Doc_Template_Mgt_filtered.json"

# 读取原始 JSON 文件
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 过滤出 db_id 为 car_1 的条目
filtered = [item for item in data if item.get("db_id") == "cre_Doc_Template_Mgt"]

# 写入新文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=4, ensure_ascii=False)

print(f"✅ 筛选完成！共找到 {len(filtered)} 条属于 cre_Doc_Template_Mgt 的记录。")
print(f"结果已保存到：{output_path}")
