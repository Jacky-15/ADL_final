import numpy as np
import pandas as pd
from langchain.vectorstores import Epsilla
from pyepsilla import vectordb
from sentence_transformers import SentenceTransformer
import subprocess
from peft import PeftModel
from typing import List
from glob import glob
from torch import cuda, bfloat16
import transformers
import torch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from transformers import StoppingCriteria, StoppingCriteriaList,pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain ,LLMChain, StuffDocumentsChain,ConversationChain,SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import json
from tqdm import tqdm
from pathlib import Path
from langchain.document_loaders import TextLoader ,JSONLoader


def set_model():
  model_id ="./Taiwan-LLM-7B-v2.0-chat"
  device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
  bnb_config = transformers.BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type='nf4',
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.float16
  )
  model_config = transformers.AutoConfig.from_pretrained(
      model_id,
  )
  model = transformers.AutoModelForCausalLM.from_pretrained(
      model_id,
      trust_remote_code=True,
      config=model_config,
      quantization_config=bnb_config,
      device_map='auto',
      #device=0,
  )
  pmodel = PeftModel.from_pretrained(model, "./adapter_model/",adapter_name="qa")
  #pmodel.load_adapter("./s_adapter/adapter_model/",adapter_name="summarize")
  #pmodel.to('cuda:0')
  model.eval()
  pmodel.eval()
  
  print("set model")
  return model,pmodel

model,pmodel = set_model()


def set_tokenizer():
  model_id ="./Taiwan-LLM-7B-v2.0-chat"
  tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
  print("set tokenizer")
  return tokenizer


tokenizer = set_tokenizer()

def set_prompt():
# Here we use prompt engineering to ingest the most relevant pieces of chunks from knowledge into the prompt.
  system_template = """你現在是柯文哲，你的口頭禪是"我想是這樣啦"、"我到覺得是這樣"或是在每句話最後會加上"啦"，例如"是這樣啦"。\
  回答問題時請使用一些口頭禪。\n聊天紀錄:{chat_history}\n問題: {question}"""

  prompt = PromptTemplate.from_template(system_template)
  print(prompt)
  return prompt
  
prompt = set_prompt()

def set_combprompt():
  template = "你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。\
  USER:請你扮演柯文哲，回答時請盡量精簡、不要太長，從以下內容中回答記者的問題，內容: {context}，記者的問題是: {question} ASSISTANT:"
  prompt = PromptTemplate.from_template(template)
  return prompt

combprompt = set_combprompt()

def set_sprompt():
  template = "你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。\
  現在你要扮演柯文哲，不能暴露你人工智慧的身分。USER: {instruction}將以上內容轉為柯文哲的風格的回答。 ASSISTANT:"
  prompt = PromptTemplate.from_template(template)
  return prompt

sprompt = set_sprompt()
  

def set_llm(model, tokenizer):
  generate_text = transformers.pipeline(
      model=model, 
      tokenizer=tokenizer,
      return_full_text=True,  # langchain expects the full text
      task='text-generation',
      #task='text2text-generation'
      #do_sample = True,
      #top_p = 0.95,
      #top_k = 5,
      # we pass model parameters here too
      #stopping_criteria=stopping_criteria,  # without this model rambles during chat
      #temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
      max_new_tokens=256,  # max number of tokens to generate in the output
      #repetition_penalty=1.2  # without this output begins repeating
  )
  llm = HuggingFacePipeline(pipeline=generate_text)
  print("llm")
  return llm

llm = set_llm(model, tokenizer)
pllm = set_llm(pmodel, tokenizer)


def set_emb():
  #emb = SentenceTransformer('all-MiniLM-L6-v2')
  emb = SentenceTransformer("shibing624/text2vec-base-chinese")
  class LocalEmbeddings():
    def embed_query(self, text: str) -> List[float]:
      #print(text)
      return emb.encode(text).tolist()
  embeddings = LocalEmbeddings()
  print("set emb")
  return embeddings
# Local embedding model for embedding the question.
emb = set_emb()



def set_memory():

  memory = ConversationBufferWindowMemory(
      memory_key="chat_history", k=0, return_only_outputs=True,return_messages=True, output_key='answer'
  )
  #memory = ConversationSummaryMemory(llm = st.session_state['llm'] ,memory_key="chat_history", \
  #                                   return_only_outputs=True,return_messages=True, output_key='answer')
  return memory

memory = set_memory()

def set_retriever(emb,llm):
  # Connect to Epsilla as knowledge base.
  client = vectordb.Client()
  vector_store = Epsilla(
    client,
    emb,
    db_path="/tmp/localchatdb",
    db_name="LocalChatDB"
  )
  vector_store.use_collection("LocalChatCollection")
  compressor = LLMChainExtractor.from_llm(llm)
  compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor,
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
  )
  return compression_retriever
  #return vector_store.as_retriever()

c_retretriever = set_retriever(emb , llm)

def set_his():
  chat = ""
  return chat
chat_history = set_his()

files = glob("./data/test/*")


chain = ConversationalRetrievalChain.from_llm(llm=pllm, #combine_docs_chain_kwargs={"prompt": prompt},\
                                              retriever=c_retretriever,\
                                              condense_question_llm = llm,
                                              #verbose=True ,\
                                              combine_docs_chain_kwargs={"prompt": combprompt},\
                                              #return_source_documents=True,\
                                              #return_generated_question=True,\
                                              memory=memory,\
                                              condense_question_prompt=prompt,
                                              )

for file in files:
    data_list = []
    data = json.loads(Path(file).read_text())
    #step = len(data)
    step = 60
    for i in tqdm(range(step)):
        temp = {}
        temp['question'] = data[i]['question']
        temp['answer'] = data[i]['answer']
        content = chain({"question":temp['question']})

        idx = content['answer'].find("ASSISTANT:")
        if idx == -1:
            start_idx = 0
        else:
            start_idx = idx + len("ASSISTANT: ")
        res = content['answer'][start_idx:]
        temp['prediction'] = res.strip()
        data_list.append(temp)

    fname = file[file.rfind('/')+1:]

    with open("./data/test_result/{ff}".format(ff=fname), 'w') as f:
        json.dump(data_list, f, indent=2,ensure_ascii=False)

