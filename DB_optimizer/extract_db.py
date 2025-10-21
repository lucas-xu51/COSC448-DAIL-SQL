# =======================================
# Unused file
# Determine which database is used in the test file 
# (it is desired that during the database optimization process, not all databases but only the ones used in the test should be optimized.)
# =======================================


def extract_databases(file_path, output_path="used_databases.txt"):
    used_dbs = set()
    # Open the input file and read line by line
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # The database name is after the last tab character in each line
            if "\t" in line:
                db_name = line.split("\t")[-1]
                used_dbs.add(db_name)

    # Save the results
    with open(output_path, "w", encoding="utf-8") as out:
        for db in sorted(used_dbs):
            out.write(db + "\n")

    print(f"✅ Extraction completed, total {len(used_dbs)} databases")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    file_path = r"C:\Users\grizz\OneDrive\Desktop\COSC448\ideas\model\DAIL-SQL\dataset\spider\dev_gold.sql"
    extract_databases(file_path)