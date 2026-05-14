# Position the number of processes specified after the --nproc_per_node flag
export PYTHONNOUSERSITE=1
CUDA_VISIBLE_DEVICES=1 python sft2.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct  \
        --batch_size 4 \
        --gradient_accumulation_steps 32 \
        --dataset lastfm \
        --prompt_path "./prompt/music.txt" \
        --logging_dir "./logs/lastfm_sft_321B_200seq/" \
        --output_dir "./output/lastfm_sft_321B_200seq/" \
        --learning_rate 1e-5 \
        --num_train_epochs 3 \
        --eval_step 0.2
