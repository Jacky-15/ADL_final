import os
import json
from tqdm.contrib.concurrent import thread_map
from openai import OpenAI

# 將sentence1 sentence2的dataset轉換為QA dataset
# 把sentence2丟給chatgpt生成對應的question形成一個QA pair

def process_file(file_info):
    filename, folder_path, folder_path_new = file_info
    file_path = os.path.join(folder_path, filename)
    file_path_new = os.path.join(folder_path_new, filename.replace('.json', '_new.json'))

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    new_data = []

    for i, item in enumerate(data):
        if i<2 or i>len(data)-3:
          continue
        if len(item['answer']) > 15 and len(item['answer']) < 150: 
            choices = 1
            prompt = "以下是柯文哲黨主席回答記者的話：'"+item['answer']+"'，請推測記者的問題(簡潔一點，80字內)是什麼，用記者的第一人稱回答我，以'記者：'開頭"

            client = OpenAI(api_key="sk-xxx")
            stream = client.chat.completions.create(
                # model="gpt-3.5-turbo-1106",
                model="gpt-4-1106-preview",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=150,
                temperature=0.7,
                n=choices,
            )

            content = stream.choices[0].message.content
            if content[:3] == "記者：":
              item['question'] = content[3:]
              # del item['topic']
              # item['question'] = item.pop('sentence 1')
              # item['answer'] = item.pop('sentence 2')
              new_data.append(item)

    with open(file_path_new, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)

def main():
    folder_path = './speech'
    folder_path_new = "./speech_gpt4"
    max_threads = 6

    file_infos = [(f, folder_path, folder_path_new) for f in os.listdir(folder_path) if f.endswith('.json')]

    results = thread_map(process_file, file_infos, max_workers=max_threads)
main()