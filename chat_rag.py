import numpy as np
import pandas as pd
from langchain.vectorstores import Epsilla
from pyepsilla import vectordb
from sentence_transformers import SentenceTransformer
import streamlit as st
from peft import PeftModel
from typing import List
from glob import glob
from torch import cuda, bfloat16
import transformers
import torch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain 
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


@st.cache_resource
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
  )
  pmodel = PeftModel.from_pretrained(model, "./adapter_model/",adapter_name="qa")
  model.eval()
  pmodel.eval()
  
  print("set model")
  return model,pmodel

st.session_state['model'],st.session_state['pmodel'] = set_model()

@st.cache_resource
def set_tokenizer():
  model_id ="./Taiwan-LLM-7B-v2.0-chat"
  tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
  print("set tokenizer")
  return tokenizer

st.session_state['tokenizer'] = set_tokenizer()

@st.cache_resource
def set_prompt():
# Here we use prompt engineering to ingest the most relevant pieces of chunks from knowledge into the prompt.
  system_template = """你現在是柯文哲，你的口頭禪是"我想是這樣啦"、"我到覺得是這樣"或是在每句話最後會加上"啦"，例如"是這樣啦"。\
  回答問題時請使用一些口頭禪。\n聊天紀錄:{chat_history}\n問題: {question}"""
  prompt = PromptTemplate.from_template(system_template)
  print(prompt)
  return prompt
st.session_state['prompt'] = set_prompt()

@st.cache_resource
def set_combprompt():
  template = "你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。\
  USER:請你扮演柯文哲，從以下內容中回答記者的問題，內容: {context}，記者的問題是: {question} ASSISTANT:"
  prompt = PromptTemplate.from_template(template)
  return prompt

st.session_state['combine_prompt'] = set_combprompt()


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
      max_new_tokens=512,  # max number of tokens to generate in the output
      #repetition_penalty=1.1  # without this output begins repeating
  )

  llm = HuggingFacePipeline(pipeline=generate_text)
  print("llm")
  return llm

st.session_state['llm'] = set_llm(st.session_state['model'], st.session_state['tokenizer'])
st.session_state['pllm'] = set_llm(st.session_state['pmodel'], st.session_state['tokenizer'])


@st.cache_resource
def set_emb():
  emb = SentenceTransformer("shibing624/text2vec-base-chinese")
  class LocalEmbeddings():
    def embed_query(self, text: str) -> List[float]:
      #print(text)
      return emb.encode(text).tolist()
  embeddings = LocalEmbeddings()
  print("set emb")
  return embeddings
# Local embedding model for embedding the question.
st.session_state['emb'] = set_emb()



@st.cache_resource
def set_memory():

  memory = ConversationBufferWindowMemory(
      memory_key="chat_history", k=0, return_only_outputs=True,return_messages=True, output_key='answer'
  )
  return memory

memory = set_memory()

@st.cache_resource
def set_retriever():
  # Connect to Epsilla as knowledge base.
  client = vectordb.Client()
  vector_store = Epsilla(
    client,
    st.session_state['emb'],
    db_path="/tmp/localchatdb",
    db_name="LocalChatDB"
  )
  vector_store.use_collection("LocalChatCollection")
  compressor = LLMChainExtractor.from_llm(st.session_state['llm'])
  compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor,
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
  )
  return compression_retriever

st.session_state['c_retretriever'] = set_retriever()


chain = ConversationalRetrievalChain.from_llm(llm=st.session_state['pllm'] ,
                                              retriever=st.session_state['c_retretriever'],
                                              condense_question_llm = st.session_state['llm'],
                                              verbose=True ,
                                              combine_docs_chain_kwargs={"prompt": st.session_state['combine_prompt']},
                                              return_source_documents=True,
                                              return_generated_question=True,
                                              memory=memory,
                                              condense_question_prompt=st.session_state['prompt'],
                                              )


# The 1st welcome message
st.title("💬 柯文哲對話機器人")
if "messages" not in st.session_state:
  st.session_state["messages"] = [{"role": "assistant", "content": "今天想聊什麼?"}]
  st.session_state['history'] = []

# A fixture of chat history
for msg in st.session_state.messages:
  st.chat_message(msg["role"]).write(msg["content"])

# Answer user question upon receiving
if question := st.chat_input():
  st.session_state.messages.append({"role": "USER", "content": question})
  st.chat_message("USER").write(question)
  
  content = chain({"question":question})

  idx = content['answer'].find("ASSISTANT:")
  if idx == -1:
    start_idx = 0
  else:
    start_idx = idx + len("ASSISTANT: ")
  res = content['answer'][start_idx:]

  msg = { 'role': 'ASSISTANT', 'content': res }
  st.session_state.messages.append(msg)
  st.chat_message("ASSISTANT").write(msg['content'])