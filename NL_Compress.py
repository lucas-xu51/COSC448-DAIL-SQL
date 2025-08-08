# Import required modules and classes
from parse_tree.component.stanzaParser import StanfordNLParser
from parse_tree.component.node_mapper_init_step import NodeMapper
from parse_tree.component.query import Query  # Only import Query class for data encapsulation
import json
import os
from tqdm import tqdm  # Import progress bar library
from pathlib import Path

def main():
    # 1. Configuration file paths (original dataset files)
    input_files = [
        {
            "path": "C:/Users/grizz/OneDrive/Desktop/COSC448/ideas/model/DAIL-SQL/dataset/cosc304/dev.json",
            "type": "dev"  # Mark as validation set
        },
        {
            "path": "C:/Users/grizz/OneDrive/Desktop/COSC448/ideas/model/DAIL-SQL/dataset/cosc304/train.json",
            "type": "train"  # Mark as training set
        }
    ]
    tokens_path = "C:/Users/grizz/OneDrive/Desktop/COSC448/ideas/model/DAIL-SQL/parse_tree/zfiles/tokens.xml"
    
    # Define output root directory (create compressed_results folder under cosc304)
    output_root = "C:/Users/grizz/OneDrive/Desktop/COSC448/ideas/model/DAIL-SQL/dataset/cosc304/compressed_results"
    Path(output_root).mkdir(parents=True, exist_ok=True)  # Ensure folder exists

    # 2. Initialize parser
    print("\n===== Initializing parser =====")
    parser = StanfordNLParser()

    # 3. Process dev and train files separately
    for file_info in input_files:
        file_path = file_info["path"]
        data_type = file_info["type"]
        output_file = os.path.join(output_root, f"{data_type}_compressed_columns.jsonl")

        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File not found - {file_path}, skipped")
            continue
        
        # Load current dataset
        print(f"\n===== Starting to process {data_type} set =====")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"✅ Successfully loaded {file_path}, total {len(data)} entries")

        # Process and write results (overwrite mode)
        success_count = 0
        with open(output_file, "w", encoding="utf-8") as f_out:
            for idx, item in enumerate(tqdm(data, desc=f"Processing {data_type} data", unit="entries")):
                # Instantiate Query object
                try:
                    query = Query(
                        raw_question=item["question"],
                        question_tokens=item["question_toks"],
                        schema_graph=None
                    )
                except KeyError as e:
                    tqdm.write(f"❌ Entry {idx+1} missing required field {e}, skipped")
                    continue

                # Generate parse tree
                try:
                    parser.parse(query)
                    if not (query.parse_tree and query.parse_tree.root):
                        tqdm.write(f"❌ Entry {idx+1} failed to generate parse tree, skipped")
                        continue
                except Exception as e:
                    tqdm.write(f"❌ Error parsing entry {idx+1}: {str(e)}, skipped")
                    continue

                # Execute phrase mapping
                try:
                    compressed_pairs = NodeMapper.phrase_process(query, tokens_path)
                    if not compressed_pairs:
                        tqdm.write(f"⚠️ Entry {idx+1} did not get valid compression results, skipped")
                        continue
                except Exception as e:
                    tqdm.write(f"❌ Error compressing entry {idx+1}: {str(e)}, skipped")
                    continue

                # Key modification: Ensure index starts from 0 (if original index starts from 1, subtract 1; if already 0, keep unchanged)
                adjusted_pairs = []
                for token, idx_in_original in compressed_pairs:
                    # Force conversion to 0-based index (handle possible 1-based index)
                    adjusted_idx = idx_in_original - 1 if idx_in_original > 0 else idx_in_original
                    adjusted_pairs.append((token, adjusted_idx))

                # Construct and write JSONL data
                json_data = {
                    "compressed_column": [pair[0] for pair in adjusted_pairs],
                    "compressed_id": [pair[1] for pair in adjusted_pairs]
                }
                f_out.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                success_count += 1

        # Output statistics for current dataset
        print(f"\n===== Finished processing {data_type} set =====")
        print(f"📊 Total entries: {len(data)}")
        print(f"✅ Successfully processed: {success_count}")
        print(f"❌ Failed/Skipped: {len(data) - success_count}")
        print(f"💾 Results saved to: {os.path.abspath(output_file)}")

    print("\n===== All datasets processed =====")
    print(f"📁 All results saved to: {os.path.abspath(output_root)}")

if __name__ == "__main__":
    main()