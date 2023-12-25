python qlora_train_v1.py \
--do_train \
--do_eval \
--model_name_or_path="../Taiwan-LLM-7B-v2.0-chat" \
\
--dataset_format="final" \
--dataset="../dataset/qa_train_v1.json" \
--validation_file='../dataset/qa_train_v1.json' \
\
--max_memory_MB="30000" \
\
--max_train_samples="3584" \
--max_eval_samples="250" \
--per_device_train_batch_size="16" \
--gradient_accumulation_steps="8" \
--max_steps="28" \
--logging_steps="2" \
--save_steps="28" \

