import os
import json

def process_file(filename, folder_path, folder_path_new):
    file_path = os.path.join(folder_path, filename)
    file_path_new = os.path.join(folder_path_new, filename.replace('.json', '_new.json'))

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    new_data = []

    for i, item in enumerate(data):
        del item['content1']
        del item['content2']
        new_data.append(item)

    with open(file_path_new, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)

def main():
    folder_path = './'
    folder_path_new = "./"
    # max_threads = 6

    file_infos = [(f, folder_path, folder_path_new) for f in os.listdir(folder_path) if f.endswith('.json')]
    for filename, folder_path, folder_path_new in file_infos:
        process_file(filename, folder_path, folder_path_new)
    # results = thread_map(process_file, file_infos, max_workers=max_threads)
main()