input_file = "dataset/quality.jsonl"
output_file = "dataset/quality_short.jsonl"
max_lines = 1000

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:
    for i, line in enumerate(infile):
        if i >= max_lines:
            break
        outfile.write(line)

print(f"✅ 已保存前 {max_lines} 条数据到 {output_file}")
