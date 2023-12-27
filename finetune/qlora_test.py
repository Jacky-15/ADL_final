import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt_v1, get_prompt_v2, get_bnb_config
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--output_file", 
        type=Path, 
        default=None, 
        help="The prediction file."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    # Load LoRA
    model = PeftModel.from_pretrained(model, args.peft_path)
    model.eval()
        
    gen_kwargs = {
        "max_new_tokens": 256,
        # "do_sample": True,
        # "top_p" : 0.95,
        # "top_k" : 5,
        # "temperature" : 0.1  # 'randomness' of outputs, 0.0 is the min and 1.0 the max   
    }


    with open(args.test_data_path, "r") as f:
        test_data = json.load(f)
    test_instructions = [get_prompt_v1(data['question']) for data in test_data]

    # batch size
    batch_size =  16
    device = "cuda:0"
    # batch
    predictions = []
    for i in tqdm(range(0, len(test_instructions), batch_size)):
        batch = test_instructions[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        # output
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        # decode
        for j, output in enumerate(outputs):
            prediction = tokenizer.decode(output, skip_special_tokens=True)[len(test_instructions[i+j]):]
            predictions.append(prediction.strip())
    json_output = [{"question":test_data[i]['question'], "answer":test_data[i]['answer'], "prediction":prediction} for i, prediction in enumerate(predictions)]

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w", encoding="utf8") as f:
            json.dump(json_output, f, ensure_ascii = False, indent=2)
            f.write("\n")  

