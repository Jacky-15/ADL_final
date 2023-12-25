# ADL-Final Project
Final Project for ADL

## Directory Structure
```shell
.
├── 光恆
│   └── README.md
├── 漢文
│   └── README.md
├── 振宗
│   └── README.md
└── README.md

```
## File Discription
- Folder `光恆` is used to process the raw data and convert it into a basic form in which the QA model can be trained.
- Folder `漢文` is used to extend the data from 光恆 with gpt4 and use it to finetune the QA model(Taiwan-LLaMa).
- Folder `振宗` is used to construct the database by incorporating data from 光恆. Additionally, it facilitates the integration of the database and the QA model from 漢文 into the UI.


## Project execution procedure 
1. Run Folder 光恆 to get processed data.
2. Run Folder 漢文 to get finetune QA model.
3. Run Folder 振宗 to get final product.

For details, please refer to the README.md file in the folder.
