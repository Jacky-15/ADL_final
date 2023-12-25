from transformers import BitsAndBytesConfig
import torch

# method 1
def get_prompt_v1(question: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。\
    USER:請你扮演柯文哲，回答記者的問題，記者的問題是'{question}' ASSISTANT:"

# method 2
def get_prompt_v2(question: str, content1: str, content2: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。\
    USER:請你扮演柯文哲，回答時請盡量精簡、不要太長，從以下內容中回答記者的問題，內容: {content1}{content2}，\
    記者的問題是: {question} ASSISTANT:"


def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    return config
