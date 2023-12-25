import os
import json

# 設置資料夾路徑
folder_path = './merge'
# folder_path = '../../../my/data'
file_output = os.path.join(folder_path, "qa_interview+gpt4v2.json")

file_merge = []
# 遍歷資料夾中的所有文件
for filename in os.listdir(folder_path):
    # 構建完整的文件路徑
    file_path = os.path.join(folder_path, filename)

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in data:
        file_merge.append(item)
with open(file_output, 'w', encoding='utf-8') as file:
    json.dump(file_merge, file, indent=4, ensure_ascii=False)

