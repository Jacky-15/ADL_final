from langchain.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.vectorstores import Epsilla
from pyepsilla import vectordb
from sentence_transformers import SentenceTransformer

from typing import List
from glob import glob

#db = vectordb.Client()
#status_code, response = db.unload_db(db_name="LocalChatDB")

# Local embedding model
#model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer("shibing624/text2vec-base-chinese")
#model = SentenceTransformer("BAAI/bge-base-zh-v1.5")


# Get list of all files in "./data/"
#files = glob("./data/transcript/*")
files = glob("./data/preprocessed/speech/*")

class LocalEmbeddings():
  def embed_documents(self, texts: List[str]) -> List[List[float]]:
    return model.encode(texts).tolist()
  
embeddings = LocalEmbeddings()
all_doc = []
for file in files:
  loader = TextLoader(file)
  documents = loader.load()
  splitted_documents = RecursiveCharacterTextSplitter(separators=["\n\n\n","\n\n", "\n","。", " ", ""],chunk_size = 220,chunk_overlap  = 0,\
                                                      length_function = len,add_start_index = True,).split_documents(documents)

  #if len(splitted_documents)>0:
    #print( splitted_documents[0].metadata)
  remove_list = []
  for i,item in enumerate(splitted_documents):
    splitted_documents[i].page_content = item.page_content.replace("\n","")
    splitted_documents[i].page_content = splitted_documents[i].page_content.replace(">>>\n","")
    if splitted_documents[i].page_content == "。" or splitted_documents[i].page_content=="":
      remove_list.append(splitted_documents[i])
      print(item)
    elif splitted_documents[i].page_content[0]== '。':
      splitted_documents[i].page_content = splitted_documents[i].page_content.replace("。","",1)
      #print(splitted_documents[i])
    if splitted_documents[i].page_content.find(">>>\n") != -1 or splitted_documents[i].page_content.find("\n") != -1:
      print(splitted_documents[i])
    #print( splitted_documents[i].page_content)
  for item in remove_list:
    splitted_documents.remove(item)
  all_doc.extend(splitted_documents)

client = vectordb.Client()
vector_store = Epsilla.from_documents(
  all_doc,
  embeddings,
  client,
  db_path="/tmp/localchatdb",
  db_name="LocalChatDB",
  collection_name="LocalChatCollection"
)
  
