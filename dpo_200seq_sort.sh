# Position the number of processes specified after the --nproc_per_node flag
export PYTHONNOUSERSITE=1
gpu1=$1;
CUDA_VISIBLE_DEVICES=$gpu1 python softmax_dpo_200seq_sort.py \
            --model_name meta-llama/Llama-3.2-1B-Instruct  \
            --resume_from_checkpoint ./output/lastfm_sft_321B_200seq \
            --batch_size 4 \
            --gradient_accumulation_steps 32 \
            --dataset lastfm \
            --prompt_path ./prompt/music.txt \
            --learning_rate 1e-5 \
            --eval_step 0.1 \
            --beta 1 \
            --neg_num 1 \
            --num_train_epochs 1 \
            --logging_dir ./logs/q25_sequence_logprob_margin_lastfm_dpo_321B_200seq_1epoch/ \
            --output_dir ./output/q25_sequence_logprob_margin_lastfm_dpo_321B_200seq_1epoch/ \
            --train_data_path ./toy_200seq/lastfm-train.json \
            --val_data_path ./toy_200seq/lastfm-val.json \
            --train_candidate_score_path ./toy_200seq/train_candidate_scores.jsonl \
            --val_candidate_score_path ./toy_200seq/val_candidate_scores.jsonl \
            --reject_sort_metric sequence_logprob_margin \
            --reject_select_mode q25 \
            --strict_check True \
            # --wandb_project wandb_proj_name \
            # --wandb_name wandb_run_name > dpo.log