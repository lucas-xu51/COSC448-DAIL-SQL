import collections
import json
import os
import re
import sqlite3

from transformers import AutoTokenizer
from utils.enums import LLM
from sql_metadata import Parser


class SqliteTable(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def get_tables(path_db):
    if not os.path.exists(path_db):
        raise RuntimeError(f"{path_db} not exists")

    # init sqlite connection
    connection = sqlite3.connect(path_db)
    cur = connection.cursor()

    # extract table information
    table_info = parse_db(path_db, cur)
    # TODO: ! add here
    table_names = get_table_names(cur=cur)

    res = list()
    for table_name in table_names:
        # schema
        schema = [_[1] for _ in cur.execute(f'PRAGMA table_info("{table_name}")')]

        # data
        data = None
        # data = cur.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchall()

        # append table
        res.append(
            SqliteTable(
                name=table_name,
                schema=schema,
                data=data,
                table_info=table_info.get(table_name, dict())
            )
        )

    cur.close()
    return res


def parse_db(path_db, cur=None):
    """Parse the sql file and extract primary and foreign keys

    :param path_file:
    :return:
    """
    table_info = dict()
    table_names = get_table_names(path_db, cur)

    for table_name in table_names:
        pks = get_primary_key(table_name, path_db,cur)
        fks = get_foreign_key(table_name, path_db, cur)

        table_info[table_name] = {
            "primary_key": pks,
            "foreign_key": fks
        }
    return table_info


def execute_query(queries, path_db=None, cur=None):
    """Execute queries and return results. Reuse cur if it's not None.

    """
    assert not (path_db is None and cur is None), "path_db and cur cannot be NoneType at the same time"

    close_in_func = False
    if cur is None:
        con = sqlite3.connect(path_db)
        cur = con.cursor()
        close_in_func = True

    if isinstance(queries, str):
        results = cur.execute(queries).fetchall()
    elif isinstance(queries, list):
        results = list()
        for query in queries:
            res = cur.execute(query).fetchall()
            results.append(res)
    else:
        raise TypeError(f"queries cannot be {type(queries)}")

    # close the connection if needed
    if close_in_func:
        con.close()

    return results

def format_foreign_key(table_name: str, res: list):
    # FROM: self key | TO: target key
    res_clean = list()
    for row in res:
        table, source, to = row[2:5]
        row_clean = f"({table_name}.{source}, {table}.{to})"
        res_clean.append(row_clean)
    return res_clean


def get_foreign_key(table_name, path_db=None, cur=None):
    res_raw = execute_query(f'PRAGMA foreign_key_list("{table_name}")', path_db, cur)
    res = format_foreign_key(table_name, res_raw)
    return res


def get_primary_key(table_name, path_db=None, cur=None):
    res_raw = execute_query(f'PRAGMA table_info("{table_name}")', path_db, cur)
    pks = list()
    for row in res_raw:
        if row[5] == 1:
            pks.append(row[1])
    return pks


def get_table_names(path_db=None, cur=None):
    """Get names of all tables within the database, and reuse cur if it's not None

    """
    table_names = execute_query(queries="SELECT name FROM sqlite_master WHERE type='table'", path_db=path_db, cur=cur)
    table_names = [_[0] for _ in table_names]
    return table_names


def filter_json(raw_response: str) -> str:
    try:
        id_s = raw_response.index("{")
        id_e = raw_response.rindex("}")
        if id_s > id_e:
            raise ValueError("Wrong json format")
        else:
            return raw_response[id_s: id_e + 1]
    except ValueError:
        raise ValueError("Wrong json format")


def cost_estimate(n_tokens: int, model):
    return LLM.costs_per_thousand[model] * n_tokens / 1000


def get_sql_for_database(path_db=None, cur=None):
    close_in_func = False
    if cur is None:
        con = sqlite3.connect(path_db)
        cur = con.cursor()
        close_in_func = True

    table_names = get_table_names(path_db, cur)

    queries = [f"SELECT sql FROM sqlite_master WHERE tbl_name='{name}'" for name in table_names]

    sqls = execute_query(queries, path_db, cur)

    if close_in_func:
        cur.close()

    return [_[0][0] for _ in sqls]

def get_filtered_schema(path_db=None, cur=None, example=None):
    """提取过滤后的数据库模式，仅保留相关表中的主键、外键和查询涉及的列"""
    close_in_func = False
    # print(example)

    if cur is None:
        con = sqlite3.connect(path_db)
        cur = con.cursor()
        close_in_func = True

    # 步骤1：获取所有表名
    table_names = get_table_names(path_db, cur)
    
    # 步骤2：构建“列索引→表索引”的映射（从示例中提取）
    column_to_table = {}
    for col_idx, table_idx in example.get("column_to_table", {}).items():
        if table_idx is not None and str(table_idx).isdigit():
            column_to_table[int(col_idx)] = int(table_idx)

    # 步骤3：从示例中提取“明确提到的表”和“明确提到的列”
    sc_link = example.get("sc_link", {})
    included_tables = set()  # 需要处理的表索引集合
    included_columns = set()  # 需要处理的列索引集合

    # 提取明确提到的表（来自q_tab_match）
    for key in sc_link.get("q_tab_match", {}):
        try:
            _, tab_idx = key.split(",")  # 解析格式如"2,0"的键
            included_tables.add(int(tab_idx))
        except (ValueError, IndexError):
            continue

    # 提取明确提到的列（来自q_col_match）
    for key in sc_link.get("q_col_match", {}):
        try:
            _, col_idx = key.split(",")  # 解析格式如"x,y"的键
            included_columns.add(int(col_idx))
        except (ValueError, IndexError):
            continue

    # 步骤4：补充“包含相关列但未被提及的表”
    missing_tables = set()
    for col_idx in included_columns:
        if col_idx in column_to_table:
            table_idx = column_to_table[col_idx]  # 列所属的表
            if table_idx not in included_tables:
                missing_tables.add(table_idx)  # 该表需被包含

    included_tables.update(missing_tables)

    # 若未指定任何表，则包含所有表
    if not included_tables:
        included_tables = set(range(len(table_names)))

    # 步骤5：通过外键关联补充相关表（确保关联完整性）
    # 收集包含表的外键查询
    fk_queries = []
    for tab_idx in included_tables:
        if tab_idx < len(table_names):
            fk_queries.append(f"PRAGMA foreign_key_list('{table_names[tab_idx]}')")
    
    fk_results = execute_query(fk_queries, path_db, cur)  # 执行外键查询
    
    # 补充外键关联的表
    for result in fk_results:
        for row in result:
            if len(row) >= 3:  # 确保外键信息完整
                ref_table = row[2]  # 外键引用的表名
                if ref_table in table_names:
                    referenced_idx = table_names.index(ref_table)
                    included_tables.add(referenced_idx)  # 加入关联表

    # 步骤6：提取每个表的关键信息（所有列、主键、外键）
    table_info = {}  # 存储每个表的详细信息：{表索引: {columns: [], pk: [], fk: []}}
    for tab_idx in included_tables:
        if tab_idx >= len(table_names):
            continue
        table_name = table_names[tab_idx]
        
        # 6.1 获取表的所有列（通过PRAGMA table_info）
        col_query = f"PRAGMA table_info('{table_name}')"
        col_result = execute_query([col_query], path_db, cur)[0]
        all_columns = [col[1] for col in col_result]  # 列名列表（索引1是列名）
        
        # 6.2 提取主键列（PRAGMA table_info中pk=1的列）
        primary_keys = [col[1] for col in col_result if col[5] == 1]  # 索引5是pk标记
        
        # 6.3 提取外键列（通过PRAGMA foreign_key_list）
        fk_query = f"PRAGMA foreign_key_list('{table_name}')"
        fk_result = execute_query([fk_query], path_db, cur)[0]
        foreign_keys = [row[3] for row in fk_result]  # 索引3是本地外键列名
        
        table_info[tab_idx] = {
            "columns": all_columns,
            "pk": primary_keys,
            "fk": foreign_keys
        }

    # 步骤7：确定每个表需要保留的列（相关列 + 主键 + 外键）
    keep_columns = {}  # {表索引: 需保留的列名集合}
    for tab_idx in included_tables:
        if tab_idx not in table_info:
            continue
        ti = table_info[tab_idx]
        table_name = table_names[tab_idx]
        
        # 7.1 提取该表的“相关列”（示例中明确提到的列）
        related_cols = set()
        for col_idx in included_columns:
            if column_to_table.get(col_idx) == tab_idx:  # 列属于当前表
                col_name = example["column_names_original"][col_idx]
                if col_name in ti["columns"]:  # 验证列存在性
                    related_cols.add(col_name)
        
        # 7.2 合并“相关列 + 主键 + 外键”（去重）
        must_keep = related_cols.union(ti["pk"]).union(ti["fk"])
        keep_columns[tab_idx] = must_keep

    # 步骤8：生成过滤后的CREATE语句（只保留必要的列和约束）
    filtered_sqls = []
    for tab_idx in included_tables:
        if tab_idx >= len(table_names):
            continue
        table_name = table_names[tab_idx]
        ti = table_info[tab_idx]
        must_keep = keep_columns[tab_idx]
        
        # 8.1 获取原CREATE语句
        create_query = f"SELECT sql FROM sqlite_master WHERE tbl_name='{table_name}'"
        create_result = execute_query([create_query], path_db, cur)[0]
        if not create_result or not create_result[0][0]:
            continue  # 跳过无CREATE语句的表
        original_create = create_result[0][0]

        # 8.2 解析原CREATE语句，过滤列定义
        # 示例原语句格式：CREATE TABLE customer (cid INT PRIMARY KEY, cname TEXT, ...)
        # 提取括号内的内容（表结构部分）
        start = original_create.find('(') + 1
        end = original_create.rfind(')')
        if start >= end:
            filtered_sqls.append(original_create)  # 解析失败时保留原语句
            continue
        struct_part = original_create[start:end].strip()
        
        # 拆分列定义和约束（按逗号分割，忽略约束中的逗号）
        parts = []
        in_constraint = False  # 标记是否在约束中（如FOREIGN KEY (...)）
        current_part = []
        for c in struct_part:
            if c == '(':
                in_constraint = True
                current_part.append(c)
            elif c == ')':
                in_constraint = False
                current_part.append(c)
            elif c == ',' and not in_constraint:
                parts.append(''.join(current_part).strip())
                current_part = []
            else:
                current_part.append(c)
        if current_part:
            parts.append(''.join(current_part).strip())
        
        # 过滤列定义：只保留需保留的列
        filtered_parts = []
        for part in parts:
            # 判断是否为列定义（格式：列名 类型 [约束]）
            if ' ' in part and not part.strip().upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE')):
                col_name = part.split()[0].strip()  # 提取列名（第一个单词）
                if col_name in must_keep:
                    filtered_parts.append(part)  # 保留需保留的列
            else:
                # 保留约束（主键、外键等，需确保涉及的列已保留）
                filtered_parts.append(part)
        
        # 重构CREATE语句
        new_struct = ', '.join(filtered_parts)
        new_create = f"{original_create[:start]}{new_struct}{original_create[end:]}"
        filtered_sqls.append(new_create)

    # print(f"包含的表索引：{included_tables}")
    # print(f"每个表保留的列：{ {table_names[k]: v for k, v in keep_columns.items()} }")

    if close_in_func:
        cur.close()

    return filtered_sqls

# def get_filtered_schema(path_db=None, cur=None, example=None):
#     """Extract filtered schema with automatic inclusion of tables containing required columns"""
#     close_in_func = False
#     if cur is None:
#         con = sqlite3.connect(path_db)
#         cur = con.cursor()
#         close_in_func = True

#     # Step 1: Get all table names and build complete schema info
#     table_names = get_table_names(path_db, cur)
    
#     # Step 2: Build column to table mapping from example
#     column_to_table = {}
#     for col_idx, table_idx in example.get("column_to_table", {}).items():
#         if table_idx is not None and str(table_idx).isdigit():
#             column_to_table[int(col_idx)] = int(table_idx)

#     # Step 3: Extract required tables and columns from example
#     sc_link = example.get("sc_link", {})
#     included_tables = set()
#     included_columns = set()

#     # Get explicitly mentioned tables
#     for key in sc_link.get("q_tab_match", {}):
#         try:
#             _, tab_idx = key.split(",")
#             included_tables.add(int(tab_idx))
#         except (ValueError, IndexError):
#             continue

#     # Get explicitly mentioned columns
#     for key in sc_link.get("q_col_match", {}):
#         try:
#             _, col_idx = key.split(",")
#             included_columns.add(int(col_idx))
#         except (ValueError, IndexError):
#             continue

#     # Step 4: Find tables containing required columns but not included
#     missing_tables = set()
#     for col_idx in included_columns:
#         if col_idx in column_to_table:
#             table_idx = column_to_table[col_idx]
#             if table_idx not in included_tables:
#                 missing_tables.add(table_idx)

#     # Step 5: Add missing tables to included_tables
#     included_tables.update(missing_tables)

#     # If no tables specified, include all tables
#     if not included_tables:
#         included_tables = set(range(len(table_names)))

#     # Step 6: Find related tables through foreign keys
#     # First get all columns from the included tables
#     queries = []
#     for tab_idx in included_tables:
#         if tab_idx < len(table_names):
#             queries.append(f"PRAGMA table_info('{table_names[tab_idx]}')")
    
#     # Execute all queries
#     execute_query(queries, path_db, cur)  # We don't need results here
    
#     # Now find foreign key relations
#     fk_queries = []
#     for tab_idx in included_tables:
#         if tab_idx < len(table_names):
#             fk_queries.append(f"PRAGMA foreign_key_list('{table_names[tab_idx]}')")
    
#     fk_results = execute_query(fk_queries, path_db, cur)
    
#     # Add referenced tables to included_tables
#     for result in fk_results:
#         for row in result:
#             if len(row) >= 3:  # Ensure row has enough elements
#                 ref_table = row[2]  # Referenced table name
#                 if ref_table in table_names:
#                     referenced_idx = table_names.index(ref_table)
#                     included_tables.add(referenced_idx)

#     # Step 7: Get CREATE statements for all included tables
#     queries = []
#     for tab_idx in included_tables:
#         if tab_idx < len(table_names):
#             queries.append(f"SELECT sql FROM sqlite_master WHERE tbl_name='{table_names[tab_idx]}'")
    
#     sqls = execute_query(queries, path_db, cur)

#     print(f"Included tables: {included_tables}")

#     if close_in_func:
#         cur.close()

#     # Filter out None or empty results
#     return [result[0][0] for result in sqls if result and result[0]]


def get_filtered_schema_with_examples(path_db=None, cur=None, example=None):
    """提取过滤后的数据库模式，为保留的列添加示例数据（支持字符串、数字、日期类型）"""
    close_in_func = False
    if cur is None:
        con = sqlite3.connect(path_db)
        cur = con.cursor()
        close_in_func = True

    # 步骤1：获取所有表名
    table_names = get_table_names(path_db, cur)  # 假设get_table_names已实现
    
    # 步骤2：构建“列索引→表索引”映射
    column_to_table = {}
    for col_idx, table_idx in example.get("column_to_table", {}).items():
        if table_idx is not None and str(table_idx).isdigit():
            column_to_table[int(col_idx)] = int(table_idx)

    # 步骤3：提取明确提到的表和列
    sc_link = example.get("sc_link", {})
    included_tables = set()
    included_columns = set()

    # 提取明确提到的表
    for key in sc_link.get("q_tab_match", {}):
        try:
            _, tab_idx = key.split(",")
            included_tables.add(int(tab_idx))
        except (ValueError, IndexError):
            continue

    # 提取明确提到的列
    for key in sc_link.get("q_col_match", {}):
        try:
            _, col_idx = key.split(",")
            included_columns.add(int(col_idx))
        except (ValueError, IndexError):
            continue

    # 步骤4：补充包含相关列的表
    missing_tables = set()
    for col_idx in included_columns:
        if col_idx in column_to_table:
            table_idx = column_to_table[col_idx]
            if table_idx not in included_tables:
                missing_tables.add(table_idx)
    included_tables.update(missing_tables)

    # 若未指定表，包含所有表
    if not included_tables:
        included_tables = set(range(len(table_names)))

    # 步骤5：通过外键关联补充表
    fk_queries = []
    for tab_idx in included_tables:
        if tab_idx < len(table_names):
            fk_queries.append(f"PRAGMA foreign_key_list('{table_names[tab_idx]}')")
    fk_results = execute_query(fk_queries, path_db, cur)  # 假设execute_query已实现
    for result in fk_results:
        for row in result:
            if len(row) >= 3 and row[2] in table_names:
                referenced_idx = table_names.index(row[2])
                included_tables.add(referenced_idx)

    # 步骤6：提取每个表的关键信息（列、主键、外键）并确定保留列
    table_info = {}  # {表索引: {columns: [], pk: [], fk: [], must_keep: []}}
    for tab_idx in included_tables:
        if tab_idx >= len(table_names):
            continue
        table_name = table_names[tab_idx]
        
        # 获取列信息
        cur.execute(f"PRAGMA table_info('{table_name}')")
        col_result = cur.fetchall()
        all_columns = [col[1] for col in col_result]
        primary_keys = [col[1] for col in col_result if col[5] == 1]  # pk标记在索引5
        
        # 获取外键列
        cur.execute(f"PRAGMA foreign_key_list('{table_name}')")
        fk_result = cur.fetchall()
        foreign_keys = [row[3] for row in fk_result]  # 本地外键列在索引3
        
        # 确定相关列（示例中提到的列）
        related_cols = set()
        for col_idx in included_columns:
            if column_to_table.get(col_idx) == tab_idx:
                col_name = example["column_names_original"][col_idx]
                if col_name in all_columns:
                    related_cols.add(col_name)
        
        # 保留列 = 相关列 + 主键 + 外键
        must_keep = related_cols.union(primary_keys).union(foreign_keys)
        table_info[tab_idx] = {
            "columns": all_columns,
            "pk": primary_keys,
            "fk": foreign_keys,
            "must_keep": must_keep
        }

    # 步骤7：生成带示例的过滤后schema
    final_schemas = []
    for tab_idx in included_tables:
        if tab_idx not in table_info:
            continue
        ti = table_info[tab_idx]
        table_name = table_names[tab_idx]
        must_keep = ti["must_keep"]
        
        # 获取原CREATE语句
        cur.execute(f"SELECT sql FROM sqlite_master WHERE tbl_name='{table_name}'")
        create_stmt = cur.fetchone()
        if not create_stmt or not create_stmt[0]:
            continue
        original_create = create_stmt[0]

        # 为保留列获取示例数据
        column_examples = {}  # {列名: [示例值]}
        for col in col_result:
            col_name = col[1]
            col_type = col[2].upper() if col[2] else ""
            if col_name not in must_keep:
                continue  # 只处理保留列
            
            # 查询非空示例（最多3个）
            try:
                cur.execute(f"SELECT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT 2")
                samples = [row[0] for row in cur.fetchall() if row[0] is not None]
                if not samples:
                    continue
            except Exception as e:
                print(f"获取示例失败 {table_name}.{col_name}: {e}")
                continue
            
            # 格式化示例（根据列类型处理）
            formatted = []
            for s in samples:
                if isinstance(s, str) or any(t in col_type for t in ["TEXT", "VARCHAR", "CHAR"]):
                    # 字符串类型：加引号并转义内部引号
                    escaped = s.replace('"', '\\"').replace("'", "\\'")
                    formatted.append(f"'{escaped}'")
                elif any(t in col_type for t in ["INT", "NUM", "DEC", "FLOAT"]):
                    # 数字类型：直接保留
                    formatted.append(str(s))
                elif any(t in col_type for t in ["DATE", "DATETIME", "TIMESTAMP"]):
                    # 日期类型：加引号
                    formatted.append(f"'{s}'")
                else:
                    # 其他类型：默认加引号
                    formatted.append(f"'{s}'")
            column_examples[col_name] = formatted

        # 修改CREATE语句，添加示例注释
        start = original_create.find('(') + 1
        end = original_create.rfind(')')
        if start >= end:
            final_schemas.append(original_create)
            continue
        struct_part = original_create[start:end].strip()
        
        # 拆分列定义和约束
        parts = []
        in_constraint = False
        current_part = []
        for c in struct_part:
            if c == '(':
                in_constraint = True
                current_part.append(c)
            elif c == ')':
                in_constraint = False
                current_part.append(c)
            elif c == ',' and not in_constraint:
                parts.append(''.join(current_part).strip())
                current_part = []
            else:
                current_part.append(c)
        if current_part:
            parts.append(''.join(current_part).strip())
        
        # 过滤并添加示例
        filtered_parts = []
        for part in parts:
            # 处理列定义（非约束）
            if ' ' in part and not part.strip().upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE')):
                col_name = part.split()[0].strip()
                if col_name not in must_keep:
                    continue  # 过滤非保留列
                # 添加示例注释
                if col_name in column_examples:
                    example_str = ", ".join(column_examples[col_name])
                    part += f"  # e.g.: {example_str}"
                filtered_parts.append(part)
            else:
                # 保留约束
                filtered_parts.append(part)
        
        # 重构CREATE语句
        new_struct = ', '.join(filtered_parts)
        new_create = f"{original_create[:start]}{new_struct}{original_create[end:]}"
        final_schemas.append(new_create)

    if close_in_func:
        cur.close()
        con.close()

    return final_schemas

# def get_filtered_schema_with_examples(path_db=None, cur=None, example=None):
#     """Extract filtered schema with automatic inclusion of tables containing required columns"""
#     close_in_func = False
#     if cur is None:
#         con = sqlite3.connect(path_db)
#         cur = con.cursor()
#         close_in_func = True

#     table_names = get_table_names(path_db, cur)
#     column_to_table = {}
#     for col_idx, table_idx in example.get("column_to_table", {}).items():
#         if table_idx is not None and str(table_idx).isdigit():
#             column_to_table[int(col_idx)] = int(table_idx)
#     sc_link = example.get("sc_link", {})
#     included_tables = set()
#     included_columns = set()
#     for key in sc_link.get("q_tab_match", {}):
#         try:
#             _, tab_idx = key.split(",")
#             included_tables.add(int(tab_idx))
#         except (ValueError, IndexError):
#             continue
#     for key in sc_link.get("q_col_match", {}):
#         try:
#             _, col_idx = key.split(",")
#             included_columns.add(int(col_idx))
#         except (ValueError, IndexError):
#             continue
#     missing_tables = set()
#     for col_idx in included_columns:
#         if col_idx in column_to_table:
#             table_idx = column_to_table[col_idx]
#             if table_idx not in included_tables:
#                 missing_tables.add(table_idx)
#     included_tables.update(missing_tables)
#     if not included_tables:
#         included_tables = set(range(len(table_names)))
#     queries = []
#     for tab_idx in included_tables:
#         if tab_idx < len(table_names):
#             queries.append(f"PRAGMA table_info('{table_names[tab_idx]}')")
#     execute_query(queries, path_db, cur)
#     fk_queries = []
#     for tab_idx in included_tables:
#         if tab_idx < len(table_names):
#             fk_queries.append(f"PRAGMA foreign_key_list('{table_names[tab_idx]}')")
#     fk_results = execute_query(fk_queries, path_db, cur)
#     for result in fk_results:
#         for row in result:
#             if len(row) >= 3:  # Ensure row has enough elements
#                 ref_table = row[2]  # Referenced table name
#                 if ref_table in table_names:
#                     referenced_idx = table_names.index(ref_table)
#                     included_tables.add(referenced_idx)

#     # Step 7: Get CREATE statements and sample data for all included tables
#     final_schemas = []
#     for tab_idx in included_tables:
#         if tab_idx >= len(table_names):
#             continue
            
#         table_name = table_names[tab_idx]
        
#         # Get CREATE statement
#         cur.execute(f"SELECT sql FROM sqlite_master WHERE tbl_name='{table_name}'")
#         create_stmt = cur.fetchone()
#         if not create_stmt or not create_stmt[0]:
#             continue
            
#         # Get column info
#         cur.execute(f"PRAGMA table_info('{table_name}')")
#         columns = cur.fetchall()
        
#         # Get sample data only for string-type columns
#         column_examples = {}
#         for col in columns:
#             col_name = col[1]
#             col_type = col[2].upper() if col[2] else ""
            
#             # Only process string-type columns
#             if any(s_type in col_type for s_type in ['TEXT', 'VARCHAR', 'CHAR', 'STRING']):
#                 try:
#                     cur.execute(f"SELECT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT 1")
#                     samples = [row[0] for row in cur.fetchall() if row[0] is not None]
                    
#                     # Format string values with quotes, others as-is
#                     formatted_samples = []
#                     for sample in samples:
#                         if isinstance(sample, str):
#                             # Escape quotes in the string
#                             escaped = sample.replace('"', '\\"')
#                             formatted_samples.append(f'"{escaped}"')
#                         else:
#                             formatted_samples.append(str(sample))
                    
#                     if formatted_samples:
#                         column_examples[col_name] = formatted_samples
#                 except Exception as e:
#                     print(f"Error getting examples for {table_name}.{col_name}: {str(e)}")
#                     continue
        
#         # Modify CREATE statement to include examples
#         modified_stmt = create_stmt[0]
#         for col in columns:
#             col_name = col[1]
#             if col_name in column_examples:
#                 example_str = ", ".join(column_examples[col_name])
#                 comment = f"  # e.g.: {example_str}"
                
#                 # Find the column definition in CREATE statement
#                 # More robust replacement to avoid partial matches
#                 col_def_pattern = re.compile(rf"\b{re.escape(col_name)}\s+[^,\n)]+")
#                 modified_stmt = col_def_pattern.sub(
#                     lambda m: m.group() + comment, 
#                     modified_stmt
#                 )
        
#         final_schemas.append(modified_stmt)
    
#     if close_in_func:
#         cur.close()
#         con.close()
    
#     return final_schemas


# def get_filtered_schema(path_db=None, cur=None, example=None):
    # """extract the filtered schema from the database based on the example provided"""
    # close_in_func = False
    # if cur is None:
    #     con = sqlite3.connect(path_db)
    #     cur = con.cursor()
    #     close_in_func = True

    # # Get all table names
    # table_names = get_table_names(path_db, cur)

    # # Extract the tables and columns to include from the example
    # sc_link = example.get("sc_link", {})
    # included_tables = set()
    # included_columns = set()


    # # Extract the table indices to include
    # for key in sc_link.get("q_tab_match", {}):
    #     _, tab_idx = key.split(",")
    #     included_tables.add(int(tab_idx))

    # # Extract the column indices to include
    # for key in sc_link.get("q_col_match", {}):
    #     _, col_idx = key.split(",")
    #     included_columns.add(int(col_idx))

    # # If no tables are specified, include all tables
    # if not included_tables:
    #     included_tables = set(range(len(table_names)))

    # # Get the complete CREATE statements for each included table
    # queries = []
    # for tab_idx in included_tables:
    #     if tab_idx < len(table_names):
    #         queries.append(f"SELECT sql FROM sqlite_master WHERE tbl_name='{table_names[tab_idx]}'")
    
    # sqls = execute_query(queries, path_db, cur)

    # # print(sqls)
    # return [_[0][0] for _ in sqls]


# def parse_create_statement(create_stmt):
#     start = create_stmt.find("(")
#     end = create_stmt.rfind(")")
#     if start == -1 or end == -1:
#         return {}
    
#     content = create_stmt[start+1:end].strip()
#     column_defs = []
#     current = ""
#     paren_level = 0
#     for c in content:
#         if c == '(':
#             paren_level += 1
#         elif c == ')':
#             paren_level -= 1
#         if c == ',' and paren_level == 0:
#             column_defs.append(current.strip())
#             current = ""
#         else:
#             current += c
#     if current:
#         column_defs.append(current.strip())
    
#     return {i: col for i, col in enumerate(column_defs)}

# def get_global_column_index(example, table_idx, column_idx):
#     table_to_columns = example.get("table_to_columns", {})
#     columns_for_table = table_to_columns.get(str(table_idx), [])
#     if column_idx < len(columns_for_table):
#         return int(columns_for_table[column_idx])
#     return -1


def get_tokenizer(tokenizer_type: str):
    return 0
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, use_fast=False)
    return tokenizer


def count_tokens(string: str, tokenizer_type: str=None, tokenizer=None):
    return 0
    # if tokenizer is None:
    #     tokenizer = get_tokenizer(tokenizer_type)
    #
    # n_tokens = len(tokenizer.encode(string))
    # return n_tokens


def sql_normalization(sql):
    sql = sql.strip()
    def white_space_fix(s):
        parsed_s = Parser(s)
        s = " ".join([token.value for token in parsed_s.tokens])

        return s

    # convert everything except text between single quotation marks to lower case
    def lower(s):
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()

            if char == "'":
                if in_quotation:
                    in_quotation = False
                else:
                    in_quotation = True

        return out_s

    # remove ";"
    def remove_semicolon(s):
        if s.endswith(";"):
            s = s[:-1]
        return s

    # double quotation -> single quotation
    def double2single(s):
        return s.replace("\"", "'")

    def add_asc(s):
        pattern = re.compile(r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")

        return s

    def sql_split(s):
        while "  " in s:
            s = s.replace("  ", " ")
        s = s.strip()
        i = 0
        toks = []
        while i < len(s):
            tok = ""
            if s[i] == "'":
                tok = tok + s[i]
                i += 1
                while i < len(s) and s[i] != "'":
                    tok = tok + s[i]
                    i += 1
                if i < len(s):
                    tok = tok + s[i]
                    i += 1
            else:
                while i < len(s) and s[i] != " ":
                    tok = tok + s[i]
                    i += 1
                while i < len(s) and s[i] == " ":
                    i += 1
            toks.append(tok)
        return toks

    def remove_table_alias(s):
        tables_aliases = Parser(s).tables_aliases
        new_tables_aliases = {}
        for i in range(1, 11):
            if "t{}".format(i) in tables_aliases.keys():
                new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]
        table_names = []
        for tok in sql_split(s):
            if '.' in tok:
                table_names.append(tok.split('.')[0])
        for table_name in table_names:
            if table_name in tables_aliases.keys():
                new_tables_aliases[table_name] = tables_aliases[table_name]
        tables_aliases = new_tables_aliases

        new_s = []
        pre_tok = ""
        for tok in sql_split(s):
            if tok in tables_aliases.keys():
                if pre_tok == 'as':
                    new_s = new_s[:-1]
                elif pre_tok != tables_aliases[tok]:
                    new_s.append(tables_aliases[tok])
            elif '.' in tok:
                split_toks = tok.split('.')
                for i in range(len(split_toks)):
                    if len(split_toks[i]) > 2 and split_toks[i][0] == "'" and split_toks[i][-1] == "'":
                        split_toks[i] = split_toks[i].replace("'", "")
                        split_toks[i] = split_toks[i].lower()
                    if split_toks[i] in tables_aliases.keys():
                        split_toks[i] = tables_aliases[split_toks[i]]
                new_s.append('.'.join(split_toks))
            else:
                new_s.append(tok)
            pre_tok = tok

        # remove as
        s = new_s
        new_s = []
        for i in range(len(s)):
            if s[i] == "as":
                continue
            if i > 0 and s[i-1] == "as":
                continue
            new_s.append(s[i])
        new_s = ' '.join(new_s)

        # for k, v in tables_aliases.items():
        #     s = s.replace("as " + k + " ", "")
        #     s = s.replace(k, v)

        return new_s

    processing_func = lambda x: remove_table_alias(add_asc(lower(white_space_fix(double2single(remove_semicolon(x))))))

    return processing_func(sql.strip())


def sql2skeleton(sql: str, db_schema):
    sql = sql_normalization(sql)

    table_names_original, table_dot_column_names_original, column_names_original = [], [], []
    column_names_original.append("*")
    for table_id, table_name_original in enumerate(db_schema["table_names_original"]):
        table_names_original.append(table_name_original.lower())
        table_dot_column_names_original.append(table_name_original + ".*")
        for column_id_and_name in db_schema["column_names_original"]:
            column_id = column_id_and_name[0]
            column_name_original = column_id_and_name[1]
            table_dot_column_names_original.append(table_name_original.lower() + "." + column_name_original.lower())
            column_names_original.append(column_name_original.lower())

    parsed_sql = Parser(sql)
    new_sql_tokens = []
    for token in parsed_sql.tokens:
        # mask table names
        if token.value in table_names_original:
            new_sql_tokens.append("_")
        # mask column names
        elif token.value in column_names_original \
                or token.value in table_dot_column_names_original:
            new_sql_tokens.append("_")
        # mask string values
        elif token.value.startswith("'") and token.value.endswith("'"):
            new_sql_tokens.append("_")
        # mask positive int number
        elif token.value.isdigit():
            new_sql_tokens.append("_")
        # mask negative int number
        elif isNegativeInt(token.value):
            new_sql_tokens.append("_")
        # mask float number
        elif isFloat(token.value):
            new_sql_tokens.append("_")
        else:
            new_sql_tokens.append(token.value.strip())

    sql_skeleton = " ".join(new_sql_tokens)

    # remove JOIN ON keywords
    sql_skeleton = sql_skeleton.replace("on _ = _ and _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace("on _ = _ or _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace(" on _ = _", "")
    pattern3 = re.compile("_ (?:join _ ?)+")
    sql_skeleton = re.sub(pattern3, "_ ", sql_skeleton)

    # "_ , _ , ..., _" -> "_"
    while ("_ , _" in sql_skeleton):
        sql_skeleton = sql_skeleton.replace("_ , _", "_")

    # remove clauses in WHERE keywords
    ops = ["=", "!=", ">", ">=", "<", "<="]
    for op in ops:
        if "_ {} _".format(op) in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("_ {} _".format(op), "_")
    while ("where _ and _" in sql_skeleton or "where _ or _" in sql_skeleton):
        if "where _ and _" in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ and _", "where _")
        if "where _ or _" in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ or _", "where _")

    # remove additional spaces in the skeleton
    while "  " in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("  ", " ")

    # double check for order by
    split_skeleton = sql_skeleton.split(" ")
    for i in range(2, len(split_skeleton)):
        if split_skeleton[i-2] == "order" and split_skeleton[i-1] == "by" and split_skeleton[i] != "_":
            split_skeleton[i] = "_"
    sql_skeleton = " ".join(split_skeleton)

    return sql_skeleton


def isNegativeInt(string):
    if string.startswith("-") and string[1:].isdigit():
        return True
    else:
        return False


def isFloat(string):
    if string.startswith("-"):
        string = string[1:]

    s = string.split(".")
    if len(s) > 2:
        return False
    else:
        for s_i in s:
            if not s_i.isdigit():
                return False
        return True


def jaccard_similarity(skeleton1, skeleton2):
    tokens1 = skeleton1.strip().split(" ")
    tokens2 = skeleton2.strip().split(" ")
    total = len(tokens1) + len(tokens2)

    def list_to_dict(tokens):
        token_dict = collections.defaultdict(int)
        for t in tokens:
            token_dict[t] += 1
        return token_dict
    token_dict1 = list_to_dict(tokens1)
    token_dict2 = list_to_dict(tokens2)

    intersection = 0
    for t in token_dict1:
        if t in token_dict2:
            intersection += min(token_dict1[t], token_dict2[t])
    union = (len(tokens1) + len(tokens2)) - intersection
    return float(intersection) / union
