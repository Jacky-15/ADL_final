import os
import re
import json
from collections import Counter

# 將記者聯訪的檔案處理為QA dataset

# output:
#   File: 20230712_1.txt
#   記者: 15
#   柯文哲: 15
#   alternating: True
def save_check_file(output_file_path, output_data):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for file_name, counts in output_data.items():
            output_file.write(f"File: {file_name}\n")
            for term, count in counts.items():
                output_file.write(f"  {term.strip()}: {count}\n")
            output_file.write("\n")

# output:
#     {
#         "question": "所以你確實有對她不滿，生氣？",
#         "answer": "我就說你們何必自作主張捏，還有一點，真的我不行，不太敢出來，那也是發言人來處理，為什麼別人來處理？好，這樣。"
#     },
def save_qa_dataset(qa_dataset, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(qa_dataset, file, ensure_ascii=False, indent=4)

#確認 兩個關鍵字 是交替出現的，用他們在文件中所有出現的位置
def is_alternating(positions1, positions2):
    combined_positions = sorted([(pos, 1) for pos in positions1] + [(pos, 2) for pos in positions2])
    for i in range(1, len(combined_positions)):
        if combined_positions[i][1] == combined_positions[i-1][1]:
            return False
    return True

# 找出記者跟柯文哲出現的位置
def find_keyword_positions(content, keyword):
    positions = []
    start = 0
    while True:
        start = content.find(keyword, start)
        if start == -1:
            break
        positions.append(start)
        start += len(keyword)
    return positions

# 確認 記者: & 柯文哲: 在文件中是交替出現的
def integrate_alternating_check(folder_keywords):
    for file_name, keywords in folder_keywords.items():
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
            content = file.read()

            if len(keywords) == 2:
                keyword1, keyword2 = keywords.keys()
                positions1 = find_keyword_positions(content, keyword1)
                positions2 = find_keyword_positions(content, keyword2)

                equal_occur = (folder_keywords[file_name][keyword1] == folder_keywords[file_name][keyword2])
                alternating = is_alternating(positions1, positions2) & equal_occur
            else:
                alternating = False  # or some default value

        folder_keywords[file_name]['alternating'] = alternating
    return folder_keywords

# 找出 換行(或文件開頭)+記者+換行 的頻繁出現keyword
def find_frequent_keywords_with_newline(file_path, min_occurrences=2, max_length=10):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    words_with_newline = re.findall(r'(?:^|\n)(\b\S+\n)', content)

    word_counts = Counter(words_with_newline)

    frequent_keywords = {word: count for word, count in word_counts.items()
                         if count >= min_occurrences and len(word) <= max_length}

    return frequent_keywords

# 找出資料夾中所有文件的keywords
def find_frequent_keywords_in_folder(folder_path, min_occurrences=2, max_length=10):
    file_keywords = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)

            keywords = find_frequent_keywords_with_newline(file_path, min_occurrences, max_length)
            file_keywords[file_name] = keywords

    return file_keywords

def create_qa_dataset(folder_path, folder_keywords):
    qa_dataset = []

    for file_name, keywords_info in folder_keywords.items():
        if len(keywords_info) != 3:
            continue
        if keywords_info.get('alternating') and len(keywords_info) == 3:
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                content = file.read()

            keywords = [key for key in keywords_info if key != 'alternating']
            positions = {key: find_keyword_positions(content, key) for key in keywords}

            combined_positions = sorted([(pos, key) for key in positions for pos in positions[key]])

            skip = False
            for i in range(0, len(combined_positions) - 1, 2):
                q_keyword, a_keyword = combined_positions[i][1], combined_positions[i + 1][1]
                q_keyword.strip()
                a_keyword.strip()
                q_start, a_start = combined_positions[i][0], combined_positions[i + 1][0]
                
                a_end = combined_positions[i + 2][0] if i + 2 < len(combined_positions) else len(content)

                question = content[q_start+len(q_keyword):a_start].replace('\n', '').strip()
                answer = content[a_start+len(a_keyword):a_end].replace('\n', '').strip()

                if "__" in question or "__" in answer:
                    continue

                if not skip:
                    qa_pair = {"question": question, "answer": answer}
                    qa_dataset.append(qa_pair)

                skip = answer.endswith("？")
    return qa_dataset

folder_path = './data'
folder_keywords = find_frequent_keywords_in_folder(folder_path)
folder_keywords_updated = integrate_alternating_check(folder_keywords)
# save_check_file('./check.txt', folder_keywords_updated)
qa_dataset = create_qa_dataset(folder_path, folder_keywords_updated)
save_qa_dataset(qa_dataset, "./qatest.json")





