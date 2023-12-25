# ADL-Final Project
Final Project for ADL

## Directory Structure
```shell
.
├── Taiwan-LLM-7B-v2.0-chat
├── adapter_model
├── chat_rag.py
├── data
│   ├── preprocessed
│   │   └── speech
│   ├── qa
│   ├── speech
│   ├── test
│   └── test_result
├── embedding.py
├── inference.py
└── speech.ipynb
```
## File Discription
- Folder `Taiwan-LLM-7B-v2.0-chat` is used to put Taiwan-LLama model.
- Folder `adapter_model` is used to put weights that we fine tune Taiwan-LLama model by qLora.
- Folder `data` is used to put all data we used.
  - Folder `preprocessed` is used to put preprocessed data.
  - Folder `qa` is used to put Question-Answer data.
  - Folder `speech` is used to put raw data we obtain from <https://hackmd.io/@johnshao>.
  - Folder `test` is used to put test data that we separate from training data.
  - Folder `test_result` is used to put preprocessed test data.
- chat_rag.py is used to construct a demo web.
- embedding.py is used to create a vector database using data in `./data/preprocessed/speech`.
- inference.py is used to create prediction data set used to rouge rating.
- speech.ipynb is used to preprocess data in `./data/speech`, the preprocessed data would be place in `./data/preprocessed/speech`

## Project execution procedure 
1. Run speech.ipynb
2. Run embedding.py. Use `python embedding.py`.
3. Run chat_rag.py to interact with the robot. Use `streamlit run chat_rag.py`.



