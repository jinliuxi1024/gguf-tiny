import json
root_dir= "train_data/主指导文件_试训练.jsonl"
with open(root_dir,encoding="utf-8") as f:
    main_guidance_data = [json.loads(line) for line in f]
converted_data = []
for entry in main_guidance_data:
    converted_entry = {
        "instruction": entry.get("prompt"),
        "input": entry.get("history"),
        "output": entry.get("completion")
    }
    converted_data.append(converted_entry)
output_file_path = 'train_data/json_data/instruct.json'
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)


print(f"转换后的数据已保存到 {output_file_path}")