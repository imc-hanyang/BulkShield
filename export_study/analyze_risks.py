import json

file_path = '/home/srt/ml_results/transformer/2026-01-25_16-04/LLM_output/llm_outputs_test_attn_en_2026-01-28_15-52.jsonl'

data = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            entry = json.loads(line)
            if 'risk_score' in entry:
                data.append(entry)
        except:
            continue

data.sort(key=lambda x: x.get('risk_score', 0), reverse=True)

# Indices: 1 (Entry 2), 3 (Entry 4), 9 (Entry 10)
indices = [1, 3, 9]

with open('selected_risks.txt', 'w', encoding='utf-8') as f:
    for i in indices:
        entry = data[i]
        f.write(f"=== Entry {i+1} (Risk: {entry.get('risk_score')}) ===\n")
        f.write(f"Test Path: {entry.get('test_path', 'N/A')}\n")
        f.write(entry.get('llm_output', 'NO OUTPUT'))
        f.write("\n\n")

print("Analysis complete. Check selected_risks.txt")
