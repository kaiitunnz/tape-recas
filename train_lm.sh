dataset="arxiv_2023"
seed=None
log_dir="logs"

WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=1 python -m core.trainLM dataset $dataset seed $seed >> ${log_dir}/output1.log 2>> ${log_dir}/error1.log &
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=1 python -m core.trainLM dataset $dataset seed $seed lm.train.use_gpt True  >> ${log_dir}/output2.log 2>> ${log_dir}/error2.log &

wait