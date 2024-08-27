def check_duplicates(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    word_dict = {}
    duplicates = []

    for idx, line in enumerate(lines):
        word = line.split('\t')[0]  # 获取词表中的词
        if word in word_dict:
            duplicates.append((word, word_dict[word], idx))
        else:
            word_dict[word] = idx

    return duplicates


# 使用示例
file_path = './spm_path/spm.vocab'
duplicates = check_duplicates(file_path)

if duplicates:
    print("发现重复项:")
    for word, first_pos, second_pos in duplicates:
        print(f"'{word}' 在位置 {first_pos + 1} 和 {second_pos + 1} 重复")
else:
    print("没有发现重复项")