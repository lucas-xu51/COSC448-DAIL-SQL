import json
from transformers import GPT2Tokenizer

INPUT_JSON_FILE = "C:\\Users\\grizz\\OneDrive\\Desktop\\COSC448\\ideas\\model\\DAIL-SQL\\dataset\\process\\SPIDER-TEST_SQLFILTER_3-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096\\questions.json"

def count_real_tokens(input_file):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        total = 0
        for i, q in enumerate(data["questions"], 1):
            prompt_text = q.get("prompt", "")
            tokens = len(tokenizer.encode(prompt_text))
            total += tokens
        return total
        
    except Exception as e:
        print("error")
        return 0

if __name__ == "__main__":
    total_tokens = count_real_tokens(INPUT_JSON_FILE)
    
    print(f"\n\ntotal tokens number: {total_tokens}")