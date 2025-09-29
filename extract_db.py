# extract_dbs.py

def extract_databases(file_path, output_path="used_databases.txt"):
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

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as out:
        for db in sorted(used_dbs):
            out.write(db + "\n")

    print(f"✅ 提取完成，一共 {len(used_dbs)} 个数据库")
    print(f"结果已保存到 {output_path}")


if __name__ == "__main__":
    file_path = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\dataset\spider\dev_gold.sql"
    extract_databases(file_path)
