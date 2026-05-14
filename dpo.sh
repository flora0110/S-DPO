# Position the number of processes specified after the --nproc_per_node flag
export PYTHONNOUSERSITE=1
CUDA_VISIBLE_DEVICES=1 python softmax_dpo2.py \
            --model_name meta-llama/Llama-3.2-1B-Instruct  \
            --resume_from_checkpoint ./output/lastfm_sft_321B \
            --batch_size 4 \
            --gradient_accumulation_steps 32 \
            --dataset lastfm \
            --prompt_path ./prompt/music.txt \
            --learning_rate 1e-5 \
            --eval_step 0.1 \
            --beta 1 \
            --neg_num 1 \
            --num_train_epochs 3 \
            --logging_dir ./logs/lastfm_dpo_321B/ \
            --output_dir ./output/lastfm_dpo_321B/
            # --wandb_project wandb_proj_name \
            # --wandb_name wandb_run_name > dpo.log