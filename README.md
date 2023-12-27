# ADL-Final Project
Final Project for ADL

## Directory Structure
```shell
.
├── Taiwan-LLM-7B-v2.0-chat
├── adapter_model_v1
├── adapter_model_v2
├── chat_rag.py
├── data
│   ├── raw_data
│   │   ├── speech_raw
│   │   ├── QA_raw
│   │   └── other_raw
│   ├── preprocessed
│   │   ├── speech
│   │   └── qa
│   ├── proc
│   ├── test
│   └── test_result
├── download.sh
├── finetune
├── embedding.py
├── inference.py
└── speech.ipynb
```
## File Discription
- Folder `Taiwan-LLM-7B-v2.0-chat` is used to put Taiwan-LLama model.
- Folder `adapter_model_v1` and `adapter_model_v2` are used to put weights that we fine tune Taiwan-LLama model by qLora.
- Folder `data` is used to put all data we used.
  - Folder `raw_data` is used to put raw data we obtain from <https://hackmd.io/@johnshao>.
  - Folder `preprocessed` is used to put preprocessed data.
  - Folder `qa` is used to put QA dataset.
  - Folder `speech` is used to put speech data we preprocessd.
  - Folder `proc` is used to put all the data preprocess source code
  - Folder `test` is used to put test data that we separate from training data.
  - Folder `test_result` is used to put inferenced test data.
- Folder `finetune` is used to put all finetuning source code.
- chat_rag.py is used to construct a demo web.
- embedding.py is used to create a vector database using data in `./data/preprocessed/speech`.
- inference.py is used to create prediction data set used to rouge rating.
- speech.ipynb is used to preprocess data in `./data/speech`, the preprocessed data would be place in `./data/preprocessed/speech`

## Preprocessing
We convert the raw data of Ko Wen-je speeches and conversations with reporters into input data for the training of the QA model, which will be further processed and then used to finetune the QA model .

1. Run QA.ipynb.
2. Run speech.ipynb.
3. Run generate_question_gpt.py. Use `python ./data/proc/generate_question_gpt.py`.

## Finetuning
1. Make sure the datas like qa_train_v1.json, qa_train_v2.json are already locate in `./data/preprocessed/qa`
2. Train the qlora model by:
   ``` shell
   sudo bash ./fintune/train_v1.sh
   ```
   v1 denote the method 1 and v2 denote the method 2.
   The peft model will locate in `./output`

## Testing
1. Download the model I trained by:
   ``` shell
   sudo bash ./download.sh
   ```
   The model will locate in  `./adapter_model_v1` and `./adapter_model_v2`
2. Test the model by:
   ``` shell
   sudo bash ./finetune/run.sh /path/to/Taiwan-Llama /path/to/adapter_checkpoint/under/your/folder /path/to/input /path/to/output
   ```
   It will produce a prediction file in /path/to/output

## Project execution procedure 
1. Run speech.ipynb.
2. Run embedding.py. Use `python embedding.py`.
3. Run chat_rag.py to interact with the robot. Use `streamlit run chat_rag.py`.
