import json
import re
import os

# ==== 路径设置 ====
dev_json_file = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\dataset\spider\dev.json"
database_folder = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\dataset\spider\database"
input_sql_file = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\DB_optimizer\RESULTS_MODEL-gpt-4.txt"
output_sql_file = os.path.splitext(input_sql_file)[0] + "_reversed_to_oldnames.txt"

# ==== 读取 dev.json ====
with open(dev_json_file, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

# ==== 缓存数据库映射 ====
mapping_cache = {}

def load_mapping(db_id):
    if db_id in mapping_cache:
        return mapping_cache[db_id]
    mapping_file = os.path.join(database_folder, db_id, f"{db_id}_name_mapping.json")
    if not os.path.exists(mapping_file):
        print(f"⚠️ 映射文件不存在: {mapping_file}")
        mapping_cache[db_id] = ({}, {})
        return {}, {}
    with open(mapping_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    table_map = {t["new_name"].lower(): t["old_name"] for t in data.get("table_mapping", [])}
    column_map = {c["column_new"].lower(): c["column_old"] for c in data.get("column_mapping", [])}
    mapping_cache[db_id] = (table_map, column_map)
    return table_map, column_map

# ==== 替换函数（返回替换次数） ====
def replace_columns(text, col_map):
    count_total = 0
    for new_col in sorted(col_map.keys(), key=len, reverse=True):
        old_col = col_map[new_col]
        pattern = re.compile(rf"\b{re.escape(new_col)}\b", re.IGNORECASE)
        matches = re.findall(pattern, text)
        count = len(matches)
        if count > 0:
            text = re.sub(pattern, old_col, text)
            count_total += count
    return text, count_total

def replace_tables(text, table_map):
    count_total = 0
    for new_table in sorted(table_map.keys(), key=len, reverse=True):
        old_table = table_map[new_table]
        pattern_check = re.compile(rf"[_]{re.escape(new_table)}|{re.escape(new_table)}[_]", re.IGNORECASE)
        if re.search(pattern_check, text):
            continue
        pattern = re.compile(rf"\b{re.escape(new_table)}\b", re.IGNORECASE)
        matches = re.findall(pattern, text)
        count = len(matches)
        if count > 0:
            text = re.sub(pattern, old_table, text)
            count_total += count
    return text, count_total

# ==== 处理每条 SQL ====
with open(input_sql_file, "r", encoding="utf-8") as f:
    sql_lines = f.readlines()

output_lines = []
total_replacements = 0

for idx, item in enumerate(dev_data):
    db_id = item["db_id"]
    sql_line = sql_lines[idx].rstrip("\n")
    table_map, column_map = load_mapping(db_id)
    
    sql_line, col_count = replace_columns(sql_line, column_map)
    sql_line, table_count = replace_tables(sql_line, table_map)
    
    total_replacements += col_count + table_count
    output_lines.append(sql_line)

# ==== 写出结果 ====
with open(output_sql_file, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print(f"\n✅ 转换完成，总共替换 {total_replacements} 处")
print(f"输出文件：{output_sql_file}")
