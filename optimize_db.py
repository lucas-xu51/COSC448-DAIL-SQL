import sqlite3
import os
import json
import openai
import shutil
from tqdm import tqdm

# -----------------------------
# 配置
# -----------------------------
# 数据库根目录 - 包含所有数据库文件夹
database_root = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\dataset\spider\database"
# 输出根目录 - 保存所有优化后的数据库和复制的未使用数据库
output_root = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\dataset\optimized_spider\database"
# dev的gold文件路径，用于提取用到的数据库
dev_gold_path = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\dataset\spider\dev_gold.sql"

# 创建输出根目录
os.makedirs(output_root, exist_ok=True)

# GPT 配置
openai.api_key = ""  # 请填入你的API密钥

# -----------------------------
# 工具函数
# -----------------------------
def extract_used_databases(file_path):
    """从dev_gold.sql中提取用到的数据库名"""
    used_dbs = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 每行最后一个制表符后面就是数据库名
            if "\t" in line:
                db_name = line.split("\t")[-1]
                used_dbs.add(db_name)
    
    print(f"✅ 提取完成，共发现 {len(used_dbs)} 个dev中用到的数据库")
    return used_dbs

def get_sqlite_and_sql_files(db_folder):
    """获取数据库文件夹中的sqlite和sql文件"""
    sqlite_file = None
    sql_file = None
    
    for file in os.listdir(db_folder):
        if file.endswith('.sqlite'):
            sqlite_file = os.path.join(db_folder, file)
        elif file.endswith('.sql'):
            sql_file = os.path.join(db_folder, file)
    
    # 检查文件名是否与文件夹名一致
    folder_name = os.path.basename(db_folder)
    if sqlite_file and os.path.splitext(os.path.basename(sqlite_file))[0] != folder_name:
        print(f"警告: {db_folder} 中的SQLite文件与文件夹名不一致")
    
    return sqlite_file, sql_file

def generate_db_summary(sqlite_file, summary_file):
    """生成数据库摘要"""
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    summary_lines = [f"Database Summary: {os.path.basename(sqlite_file)}\n"]
    db_info = {}
    
    for table in tables:
        summary_lines.append(f"Table: {table}")
        summary_lines.append("-" * (7 + len(table)))
        
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        col_names = [col[1] for col in columns]
        
        summary_lines.append("Schema:")
        for col in columns:
            summary_lines.append(f"  {col[1]} ({col[2]})")
        
        cursor.execute(f"SELECT * FROM {table} LIMIT 5;")
        rows = cursor.fetchall()
        
        if rows:
            summary_lines.append("Sample Data (first 5 rows):")
            summary_lines.append(" | ".join(col_names))
            summary_lines.append("-" * (len(" | ".join(col_names)) + 5))
            for row in rows:
                summary_lines.append(" | ".join(str(x) if x is not None else "NULL" for x in row))
        else:
            summary_lines.append("No data available.")
        
        summary_lines.append("")  # 空行
        db_info[table] = {"columns": [col[1] for col in columns]}
    
    conn.close()
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    
    return db_info

def get_optimized_mapping(summary_file):
    """调用GPT获取优化映射"""
    with open(summary_file, "r", encoding="utf-8") as f:
        db_summary_text = f.read()
    
    prompt = f"""
    Database Summary:
    {db_summary_text}
    
    You are a database expert. I am providing the database schema and sample data. 
    Some table names or column names may not clearly describe the data they contain (especially table names that are ambiguous). 
    Please optimize the names to reduce ambiguity and improve clarity. Focus on **semantic improvements** rather than just concatenating or changing the style of the names. 
    Only return a **pure JSON object** in the following format (Do not add any Markdown formatting like ```json. Only output the raw JSON object.):
    
    {{"tables": [{{"old_name": "...", "new_name": "...", "columns": [{{"old_name": "...", "new_name": "..."}}, ...]}}, ...]}}
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        optimized_mapping = response.choices[0].message.content
        return json.loads(optimized_mapping)
    except Exception as e:
        print(f"获取优化映射时出错: {str(e)}")
        return None

def optimize_database(sqlite_file, output_folder, db_name, optimized_map):
    """优化数据库表名和列名，保持原文件名不变"""
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取原始文件名（不修改名称）
    sqlite_filename = os.path.basename(sqlite_file)
    sql_filename = f"{os.path.splitext(sqlite_filename)[0]}.sql"
    
    # 新数据库文件路径 - 使用原始文件名
    new_sqlite_file = os.path.join(output_folder, sqlite_filename)
    new_sql_file = os.path.join(output_folder, sql_filename)
    mapping_file = os.path.join(output_folder, f"{db_name}_name_mapping.json")
    error_log_file = os.path.join(output_folder, f"{db_name}_errors.log")
    
    # 初始化错误日志
    errors = []
    
    try:
        # 拷贝原数据到新数据库
        shutil.copy(sqlite_file, new_sqlite_file)
        
        # 连接新数据库并设置适当的文本处理方式
        new_conn = sqlite3.connect(new_sqlite_file)
        new_conn.text_factory = lambda x: x.decode('utf-8', errors='replace') if x else x
        new_cursor = new_conn.cursor()
        
        # 获取所有现有表名
        new_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [row[0] for row in new_cursor.fetchall()]
        
        # 先收集所有表名更改，检查是否有冲突
        table_renames = []
        if optimized_map and "tables" in optimized_map:
            for table in optimized_map["tables"]:
                old_table = table["old_name"]
                new_table = table["new_name"]
                
                if old_table != new_table:
                    # 检查目标表名是否已存在
                    if new_table in existing_tables and new_table != old_table:
                        # 如果存在冲突，生成一个唯一的表名
                        original_new_table = new_table
                        counter = 1
                        while new_table in existing_tables:
                            new_table = f"{original_new_table}_{counter}"
                            counter += 1
                        error_msg = f"表名冲突，将 {old_table} 重命名为 {new_table} 而非 {original_new_table}"
                        print(f"⚠️ {error_msg}")
                        errors.append(error_msg)
                    
                    table_renames.append((old_table, new_table))
                    # 更新现有表名列表，以便后续检查
                    if old_table in existing_tables:
                        existing_tables.remove(old_table)
                        existing_tables.append(new_table)
        
        # 执行表重命名
        rename_log = []
        for old_table, new_table in table_renames:
            try:
                new_cursor.execute(f'ALTER TABLE "{old_table}" RENAME TO "{new_table}";')
                rename_log.append(f'表: "{old_table}" -> "{new_table}"')
            except sqlite3.OperationalError as e:
                error_msg = f'无法重命名表 {old_table} 到 {new_table}: {str(e)}'
                rename_log.append(f'⚠️ {error_msg}')
                errors.append(error_msg)
        
        # 重命名列
        column_renames = []
        if optimized_map and "tables" in optimized_map:
            for table in optimized_map["tables"]:
                old_table = table["old_name"]
                # 找到新表名，如果没有重命名则使用旧表名
                new_table = next((nt for ot, nt in table_renames if ot == old_table), old_table)
                
                if "columns" in table:
                    for col in table["columns"]:
                        old_col = col["old_name"]
                        new_col = col["new_name"]
                        if old_col != new_col:
                            try:
                                new_cursor.execute(f"PRAGMA table_info('{new_table}')")
                                existing_cols = [c[1] for c in new_cursor.fetchall()]
                                
                                if new_col in existing_cols:
                                    error_msg = f'跳过列 {new_table}."{old_col}" 的重命名，目标列名 "{new_col}" 已存在'
                                    rename_log.append(f'⚠️ {error_msg}')
                                    errors.append(error_msg)
                                    column_renames.append((old_table, new_table, old_col, old_col))
                                else:
                                    new_cursor.execute(f'ALTER TABLE "{new_table}" RENAME COLUMN "{old_col}" TO "{new_col}";')
                                    rename_log.append(f'列: {new_table}."{old_col}" -> "{new_col}"')
                                    column_renames.append((old_table, new_table, old_col, new_col))
                            except sqlite3.OperationalError as e:
                                error_msg = f'无法重命名列 {new_table}."{old_col}" 到 "{new_col}": {str(e)}'
                                rename_log.append(f'⚠️ {error_msg}')
                                errors.append(error_msg)
                                column_renames.append((old_table, new_table, old_col, old_col))
                        else:
                            column_renames.append((old_table, new_table, old_col, new_col))
        
        new_conn.commit()
        new_conn.close()
        
        # 保存映射关系到JSON文件
        mapping = {
            "table_mapping": [{"old_name": ot, "new_name": nt} for ot, nt in table_renames],
            "all_tables": [{"old_name": table["old_name"], 
                           "new_name": next((nt for ot, nt in table_renames if ot == table["old_name"]), table["old_name"])} 
                          for table in optimized_map["tables"]] if optimized_map and "tables" in optimized_map else [],
            "column_mapping": [{"table_old": ot, "table_new": nt, "column_old": oc, "column_new": nc} 
                              for ot, nt, oc, nc in column_renames]
        }
        
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        
        # 生成新的SQL文件（使用原始文件名，增强编码处理）
        try:
            new_conn = sqlite3.connect(new_sqlite_file)
            # 处理特殊字符
            new_conn.text_factory = lambda x: str(x, 'utf-8', 'replace')
            
            with open(new_sql_file, "w", encoding="utf-8", errors="replace") as f:
                for line in new_conn.iterdump():
                    # 确保每行都能正确编码
                    try:
                        line = line.encode('utf-8', errors='replace').decode('utf-8')
                    except UnicodeError:
                        pass
                    f.write(f"{line}\n")
            new_conn.close()
        except sqlite3.OperationalError as e:
            error_msg = f"生成SQL文件时出错: {str(e)}"
            print(f"⚠️ {error_msg}")
            errors.append(error_msg)
            # 创建一个标记错误的SQL文件
            with open(new_sql_file, "w", encoding="utf-8") as f:
                f.write(f"-- 生成SQL文件时出错: {str(e)}\n")
        
        # 保存错误日志
        if errors:
            with open(error_log_file, "w", encoding="utf-8") as f:
                f.write("数据库优化过程中出现以下错误：\n")
                f.write("\n".join(errors))
        
        return {
            "new_sqlite": new_sqlite_file,
            "new_sql": new_sql_file,
            "mapping": mapping_file,
            "log": rename_log,
            "errors": errors
        }
        
    except Exception as e:
        error_msg = f"优化数据库时发生致命错误: {str(e)}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)
        
        # 保存错误日志
        with open(error_log_file, "w", encoding="utf-8") as f:
            f.write("数据库优化过程中出现致命错误：\n")
            f.write("\n".join(errors))
            
        return {
            "new_sqlite": new_sqlite_file if os.path.exists(new_sqlite_file) else None,
            "new_sql": new_sql_file if os.path.exists(new_sql_file) else None,
            "mapping": mapping_file if os.path.exists(mapping_file) else None,
            "log": [],
            "errors": errors
        }


def copy_unused_database(db_folder, output_root):
    """复制未使用的数据库到输出目录"""
    source_path = os.path.join(database_root, db_folder)
    dest_path = os.path.join(output_root, db_folder)
    
    # 如果目标路径已存在，先删除
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    
    # 复制整个文件夹
    shutil.copytree(source_path, dest_path)
    print(f"✅ 已复制未使用的数据库: {db_folder} 到 {dest_path}")

# -----------------------------
# 主程序 - 优化dev用到的数据库，复制其他所有数据库
# -----------------------------
def main():
    # 提取dev中用到的数据库
    used_dbs = extract_used_databases(dev_gold_path)
    if not used_dbs:
        print("没有需要处理的数据库")
        return
    
    # 获取所有数据库文件夹
    all_db_folders = [f for f in os.listdir(database_root) 
                     if os.path.isdir(os.path.join(database_root, f))]
    
    # 筛选出需要处理的数据库和需要复制的数据库
    db_folders_to_process = [db for db in all_db_folders if db in used_dbs]
    db_folders_to_copy = [db for db in all_db_folders if db not in used_dbs]
    
    print(f"需要优化的数据库数量: {len(db_folders_to_process)}")
    print(f"需要复制的未使用数据库数量: {len(db_folders_to_copy)}")
    
    # 先复制所有未使用的数据库
    print("\n开始复制未使用的数据库...")
    for db_folder in tqdm(db_folders_to_copy, desc="复制未使用数据库"):
        copy_unused_database(db_folder, output_root)
    
    # 然后处理需要优化的数据库
    print("\n开始优化dev中用到的数据库...")
    for db_folder in tqdm(db_folders_to_process, desc="优化数据库"):
        db_path = os.path.join(database_root, db_folder)
        output_folder = os.path.join(output_root, db_folder)
        
        # 创建数据库特定的输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取SQLite和SQL文件
        sqlite_file, sql_file = get_sqlite_and_sql_files(db_path)
        
        if not sqlite_file:
            print(f"跳过 {db_folder} - 未找到SQLite文件")
            continue
        
        print(f"\n处理数据库: {db_folder}")
        print(f"SQLite文件: {sqlite_file}")
        
        # 1. 生成数据库摘要
        summary_file = os.path.join(output_folder, f"{db_folder}_summary.txt")
        generate_db_summary(sqlite_file, summary_file)
        print(f"✅ 数据库摘要已生成: {summary_file}")
        
        # 2. 获取优化映射
        optimized_map = get_optimized_mapping(summary_file)
        if not optimized_map:
            print(f"❌ 获取优化映射失败，跳过 {db_folder}")
            continue
        print("✅ GPT 优化映射已生成")
        
        # 3. 优化数据库
        result = optimize_database(sqlite_file, output_folder, db_folder, optimized_map)
        
        # 4. 输出结果
        print(f"✅ 优化完成: {result['new_sqlite']}")
        print(f"✅ 映射文件: {result['mapping']}")
        print(f"✅ SQL文件: {result['new_sql']}")
    
    print("\n所有操作完成!")
    print(f"优化后的数据库和复制的数据库已保存到: {output_root}")

if __name__ == "__main__":
    main()
    