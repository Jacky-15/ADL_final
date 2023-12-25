import os

# 設置資料夾路徑
folder_path = './speech'

# 遍歷資料夾中的所有文件
for filename in os.listdir(folder_path):
    # 構建完整的文件路徑
    old_file = os.path.join(folder_path, filename)

    # 檢查是否為文件以及是否沒有副檔名
    if os.path.isfile(old_file) and '.' not in filename:
        # 新的文件名（添加.json副檔名）
        new_file = old_file + '.json'
        # 重命名文件
        os.rename(old_file, new_file)
        print(f"Renamed '{old_file}' to '{new_file}'")
