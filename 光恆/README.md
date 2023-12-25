# ADL-Final Project
Final Project for ADL

## Directory Structure
```shell
.
├── data
│   ├── QA
│   ├── speech
│   └── other
├── preprocessed data
│   ├── QA
│   └── speech
├── QA.ipynb
└── speech.ipynb

## File Discription
- Folder `data` is used to put raw data from <https://hackmd.io/@johnshao>.
    - Folder `QA` is used to put put raw Question-Answer data.
    - Folder `speech` is used to put raw speech data.
    - Folder `other` is used to put other type raw data.
- Folder `preprocessed data` is used to put preprocessed data.
    - Folder `QA` is used to put put Question-Answer data we processed.
    - Folder `speech` is used to put speech data we processed.
- speech.ipynb is used to preprocess data in `./data/speech`, the preprocessed data would be place in `./data/preprocessed/speech`
- speech.ipynb is used to preprocess data in `./data/QA`, the preprocessed data would be place in `./data/preprocessed/QA`

## Project execution procedure 
In my part, I convert the raw data of 柯文哲 speeches and conversations with reporters into input data for the training of the QA model, which will be further processed and then used to fine tune the QA model .(見漢文部分)

Run QA.ipynb
Run speech.ipynb
