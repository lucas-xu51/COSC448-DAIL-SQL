# ========================================
# Optimize the spider database
# Using GPT-4 for optimization, only the dataset used in the "test" was selected for optimization, while a new database that was not used before was also copied. 
# create a new "spider database" folder with "optimied test-used db + original other db".
# ========================================

import sqlite3
import os
import json
import openai
import shutil
from tqdm import tqdm
from dotenv import load_dotenv

env_path = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\.env"
load_dotenv(env_path)


# ----------------------------- 
# Configuration
# -----------------------------
# Database root directory - contains all database folders
database_root = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\dataset\spider_old\database"
# Output root directory - stores all optimized databases and copied unused databases
output_root = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\dataset\spider\database"
# Path to dev gold file, used to extract databases in use
dev_gold_path = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\dataset\spider_old\dev_gold.sql"

# Create output root directory (ignore if exists)
os.makedirs(output_root, exist_ok=True)

# GPT Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Utility Functions
# -----------------------------
def extract_used_databases(file_path):
    """Extract names of databases used from dev_gold.sql"""
    used_dbs = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # The database name is after the last tab character in each line
            if "\t" in line:
                db_name = line.split("\t")[-1]
                used_dbs.add(db_name)
    
    print(f"✅ Extraction completed. Found {len(used_dbs)} databases used in dev environment")
    return used_dbs

def get_sqlite_and_sql_files(db_folder):
    """Get SQLite and SQL files from the database folder"""
    sqlite_file = None
    sql_file = None
    
    for file in os.listdir(db_folder):
        if file.endswith('.sqlite'):
            sqlite_file = os.path.join(db_folder, file)
        elif file.endswith('.sql'):
            sql_file = os.path.join(db_folder, file)
    
    # Check if file name matches folder name
    folder_name = os.path.basename(db_folder)
    if sqlite_file and os.path.splitext(os.path.basename(sqlite_file))[0] != folder_name:
        print(f"Warning: SQLite file in {db_folder} does not match folder name")
    
    return sqlite_file, sql_file

def generate_db_summary(sqlite_file, summary_file):
    """Generate database summary"""
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
        
        summary_lines.append("")  # Empty line for separation
        db_info[table] = {"columns": [col[1] for col in columns]}
    
    conn.close()
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    
    return db_info

def get_optimized_mapping(summary_file):
    """Call GPT to get optimized name mapping"""
    with open(summary_file, "r", encoding="utf-8") as f:
        db_summary_text = f.read()
    
    prompt = f"""
        Database Summary:
        {db_summary_text}

        You are a database and schema design expert. I am providing the database schema and some sample data.
        Some table or column names may be unclear, ambiguous, or not accurately describe the data they contain.

        Your task is to **optimize table and column names** to make them clearer and more semantically precise, while keeping the meaning consistent with the actual data.

        Follow these strict naming rules:
        1. **Avoid over-generalization.**
        Do not replace a specific concept with a broader one.
        Example: if a table represents "employees", do NOT rename it to "persons".
        Keep the name as specific as its data meaning.

        2. **Ensure every column name is unique across the database.**
        If multiple tables contain similar column names like "id" or "name",
        make each unique by prefixing the related table name.
        For example:
        - "id" → "employee_id"
        - "name" → "department_name"

        3. **Add `_fk` suffix to all foreign key columns.**
        For example:
        - "customer_id" in the "orders" table → "customer_id_fk"
        - If the name already ends with `_fk`, keep it.

        Output only a **pure JSON object** (no Markdown, no code blocks), in the format below:

        {{
        "tables": [
            {{
            "old_name": "...",
            "new_name": "...",
            "columns": [
                {{"old_name": "...", "new_name": "..."}},
                ...
            ]
            }},
            ...
        ]
        }}
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
        print(f"Error getting optimized mapping: {str(e)}")
        return None

def optimize_database(sqlite_file, output_folder, db_name, optimized_map):
    """Optimize database table and column names, keep original file names unchanged"""
    # Create output folder (ignore if exists)
    os.makedirs(output_folder, exist_ok=True)
    
    # Get original file names (no modification)
    sqlite_filename = os.path.basename(sqlite_file)
    sql_filename = f"{os.path.splitext(sqlite_filename)[0]}.sql"
    
    # Paths for new database files - use original file names
    new_sqlite_file = os.path.join(output_folder, sqlite_filename)
    new_sql_file = os.path.join(output_folder, sql_filename)
    mapping_file = os.path.join(output_folder, f"{db_name}_name_mapping.json")
    error_log_file = os.path.join(output_folder, f"{db_name}_errors.log")
    
    # Initialize error log
    errors = []
    
    try:
        # Copy original data to new database file
        shutil.copy(sqlite_file, new_sqlite_file)
        
        # Connect to new database and set appropriate text handling
        new_conn = sqlite3.connect(new_sqlite_file)
        new_conn.text_factory = lambda x: x.decode('utf-8', errors='replace') if x else x
        new_cursor = new_conn.cursor()
        
        # Get all existing table names
        new_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [row[0] for row in new_cursor.fetchall()]
        
        # Collect all table name changes first and check for conflicts
        table_renames = []
        if optimized_map and "tables" in optimized_map:
            for table in optimized_map["tables"]:
                old_table = table["old_name"]
                new_table = table["new_name"]
                
                if old_table != new_table:
                    # Check if target table name already exists
                    if new_table in existing_tables and new_table != old_table:
                        # Generate unique table name if conflict exists
                        original_new_table = new_table
                        counter = 1
                        while new_table in existing_tables:
                            new_table = f"{original_new_table}_{counter}"
                            counter += 1
                        error_msg = f"Table name conflict. Renaming {old_table} to {new_table} instead of {original_new_table}"
                        print(f"⚠️ {error_msg}")
                        errors.append(error_msg)
                    
                    table_renames.append((old_table, new_table))
                    # Update existing tables list for subsequent checks
                    if old_table in existing_tables:
                        existing_tables.remove(old_table)
                        existing_tables.append(new_table)
        
        # Execute table renaming
        rename_log = []
        for old_table, new_table in table_renames:
            try:
                new_cursor.execute(f'ALTER TABLE "{old_table}" RENAME TO "{new_table}";')
                rename_log.append(f'Table: "{old_table}" -> "{new_table}"')
            except sqlite3.OperationalError as e:
                error_msg = f'Failed to rename table {old_table} to {new_table}: {str(e)}'
                rename_log.append(f'⚠️ {error_msg}')
                errors.append(error_msg)
        
        # Rename columns
        column_renames = []
        if optimized_map and "tables" in optimized_map:
            for table in optimized_map["tables"]:
                old_table = table["old_name"]
                # Find new table name (use old name if no renaming occurred)
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
                                    error_msg = f'Skipping renaming of column {new_table}."{old_col}" - target column name "{new_col}" already exists'
                                    rename_log.append(f'⚠️ {error_msg}')
                                    errors.append(error_msg)
                                    column_renames.append((old_table, new_table, old_col, old_col))
                                else:
                                    new_cursor.execute(f'ALTER TABLE "{new_table}" RENAME COLUMN "{old_col}" TO "{new_col}";')
                                    rename_log.append(f'Column: {new_table}."{old_col}" -> "{new_col}"')
                                    column_renames.append((old_table, new_table, old_col, new_col))
                            except sqlite3.OperationalError as e:
                                error_msg = f'Failed to rename column {new_table}."{old_col}" to "{new_col}": {str(e)}'
                                rename_log.append(f'⚠️ {error_msg}')
                                errors.append(error_msg)
                                column_renames.append((old_table, new_table, old_col, old_col))
                        else:
                            column_renames.append((old_table, new_table, old_col, new_col))
        
        new_conn.commit()
        new_conn.close()
        
        # Save mapping relationship to JSON file
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
        
        # Generate new SQL file (use original file name, enhance encoding handling)
        try:
            new_conn = sqlite3.connect(new_sqlite_file)
            # Handle special characters
            new_conn.text_factory = lambda x: str(x, 'utf-8', 'replace')
            
            with open(new_sql_file, "w", encoding="utf-8", errors="replace") as f:
                for line in new_conn.iterdump():
                    # Ensure each line is encoded correctly
                    try:
                        line = line.encode('utf-8', errors='replace').decode('utf-8')
                    except UnicodeError:
                        pass
                    f.write(f"{line}\n")
            new_conn.close()
        except sqlite3.OperationalError as e:
            error_msg = f"Error generating SQL file: {str(e)}"
            print(f"⚠️ {error_msg}")
            errors.append(error_msg)
            # Create a SQL file marking the error
            with open(new_sql_file, "w", encoding="utf-8") as f:
                f.write(f"-- Error generating SQL file: {str(e)}\n")
        
        # Save error log if there are errors
        if errors:
            with open(error_log_file, "w", encoding="utf-8") as f:
                f.write("The following errors occurred during database optimization:\n")
                f.write("\n".join(errors))
        
        return {
            "new_sqlite": new_sqlite_file,
            "new_sql": new_sql_file,
            "mapping": mapping_file,
            "log": rename_log,
            "errors": errors
        }
        
    except Exception as e:
        error_msg = f"Critical error during database optimization: {str(e)}"
        print(f"❌ {error_msg}")
        errors.append(error_msg)
        
        # Save error log
        with open(error_log_file, "w", encoding="utf-8") as f:
            f.write("Critical error occurred during database optimization:\n")
            f.write("\n".join(errors))
            
        return {
            "new_sqlite": new_sqlite_file if os.path.exists(new_sqlite_file) else None,
            "new_sql": new_sql_file if os.path.exists(new_sql_file) else None,
            "mapping": mapping_file if os.path.exists(mapping_file) else None,
            "log": [],
            "errors": errors
        }


def copy_unused_database(db_folder, output_root):
    """Copy unused databases to output directory"""
    source_path = os.path.join(database_root, db_folder)
    dest_path = os.path.join(output_root, db_folder)
    
    # Delete destination path if it already exists
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    
    # Copy entire folder
    shutil.copytree(source_path, dest_path)
    print(f"✅ Copied unused database: {db_folder} to {dest_path}")

# -----------------------------
# Main Program - Optimize databases used in dev, copy all other databases
# -----------------------------
def main():
    # Extract databases used in dev environment
    used_dbs = extract_used_databases(dev_gold_path)
    if not used_dbs:
        print("No databases need to be processed")
        return
    
    # Get all database folders
    all_db_folders = [f for f in os.listdir(database_root) 
                     if os.path.isdir(os.path.join(database_root, f))]
    
    # Filter databases to be optimized and databases to be copied
    db_folders_to_process = [db for db in all_db_folders if db in used_dbs]
    db_folders_to_copy = [db for db in all_db_folders if db not in used_dbs]
    
    print(f"Number of databases to optimize: {len(db_folders_to_process)}")
    print(f"Number of unused databases to copy: {len(db_folders_to_copy)}")
    
    # First copy all unused databases
    print("\nStarting to copy unused databases...")
    for db_folder in tqdm(db_folders_to_copy, desc="Copying unused databases"):
        copy_unused_database(db_folder, output_root)
    
    # Then process databases that need optimization
    print("\nStarting to optimize databases used in dev...")
    for db_folder in tqdm(db_folders_to_process, desc="Optimizing databases"):
        db_path = os.path.join(database_root, db_folder)
        output_folder = os.path.join(output_root, db_folder)
        
        # Create database-specific output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Get SQLite and SQL files
        sqlite_file, sql_file = get_sqlite_and_sql_files(db_path)
        
        if not sqlite_file:
            print(f"Skipping {db_folder} - No SQLite file found")
            continue
        
        print(f"\nProcessing database: {db_folder}")
        print(f"SQLite file: {sqlite_file}")
        
        # 1. Generate database summary
        summary_file = os.path.join(output_folder, f"{db_folder}_summary.txt")
        generate_db_summary(sqlite_file, summary_file)
        print(f"✅ Database summary generated: {summary_file}")
        
        # 2. Get optimized mapping
        optimized_map = get_optimized_mapping(summary_file)
        if not optimized_map:
            print(f"❌ Failed to get optimized mapping, skipping {db_folder}")
            continue
        print("✅ GPT optimized mapping generated")
        
        # 3. Optimize database
        result = optimize_database(sqlite_file, output_folder, db_folder, optimized_map)
        
        # 4. Output results
        print(f"✅ Optimization completed: {result['new_sqlite']}")
        print(f"✅ Mapping file: {result['mapping']}")
        print(f"✅ SQL file: {result['new_sql']}")
    
    print("\nAll operations completed!")
    print(f"Optimized databases and copied databases are saved to: {output_root}")

if __name__ == "__main__":
    main()