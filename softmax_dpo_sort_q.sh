# Position the number of processes specified after the --nproc_per_node flag
export PYTHONNOUSERSITE=1
gpu1=$1;
reject_sort_metric=$2


base_model=meta-llama/Llama-3.2-1B-Instruct
resume_from_checkpoint=./output/lastfm_sft_321B
train_data_path=./sample_data/lastfm-sft-cans20/lastfm-train.json
val_data_path=./sample_data/lastfm-sft-cans20/lastfm-val.json
train_candidate_score_path=./sample_data/lastfm-sft-cans20/train_candidate_scores_w_hidden.jsonl
val_candidate_score_path=./sample_data/lastfm-sft-cans20/val_candidate_scores_w_hidden.jsonl
batch_size=4
gradient_accumulation_steps=32
learning_rate=1e-5
eval_step=0.1
beta=1
neg_num=3
num_train_epochs=3
# sigma_ratio=0.25
reject_select_mode="bell_random"
center_q=0.0
for sigma_ratio in 0.15 0.05
do
    logging_dir=./logs/sdpo_neg3/${reject_sort_metric}_${reject_select_mode}_p${center_q}_sigma${sigma_ratio}_lastfm_321B/
    output_dir=./output/sdpo_neg3/${reject_sort_metric}_${reject_select_mode}_p${center_q}_sigma${sigma_ratio}_lastfm_321B/
    
    mkdir -p "$logging_dir"
    mkdir -p "$output_dir"

    echo_log_file=${logging_dir}/run_info.log
    python_log_file=${logging_dir}/dpo.log


    {
        echo ----------------- Settings for dpo training -----------------
        echo "GPU: $gpu1"
        echo "base_model: $base_model"
        echo "resume_from_checkpoint: $resume_from_checkpoint"
        echo "train_data_path: $train_data_path"
        echo "val_data_path: $val_data_path"
        echo "train_candidate_score_path: $train_candidate_score_path"
        echo "val_candidate_score_path: $val_candidate_score_path"
        echo "batch_size: $batch_size"
        echo "gradient_accumulation_steps: $gradient_accumulation_steps"
        echo "learning_rate: $learning_rate"
        echo "eval_step: $eval_step"
        echo "beta: $beta"
        echo "neg_num: $neg_num"
        echo "num_train_epochs: $num_train_epochs"
        echo "sigma_ratio: $sigma_ratio"
        echo ---------------------------------------------------------------

        echo ----------------- DPO Training Mode -----------------
        echo "Reject select mode: $reject_select_mode"
        echo "Reject sort metric: $reject_sort_metric"
        echo "logging_dir: $logging_dir"
        echo "output_dir: $output_dir"
        echo -----------------------------------------------------
    } | tee "$echo_log_file"
    
    {
        CUDA_VISIBLE_DEVICES=$gpu1 python softmax_dpo_sort_q.py \
                    --model_name $base_model  \
                    --resume_from_checkpoint $resume_from_checkpoint \
                    --batch_size $batch_size \
                    --gradient_accumulation_steps $gradient_accumulation_steps \
                    --dataset lastfm \
                    --prompt_path ./prompt/music.txt \
                    --learning_rate $learning_rate \
                    --eval_step $eval_step \
                    --beta $beta \
                    --neg_num $neg_num \
                    --num_train_epochs $num_train_epochs \
                    --logging_dir $logging_dir \
                    --output_dir $output_dir \
                    --train_data_path $train_data_path \
                    --val_data_path $val_data_path \
                    --train_candidate_score_path $train_candidate_score_path \
                    --val_candidate_score_path $val_candidate_score_path \
                    --reject_sort_metric $reject_sort_metric \
                    --reject_select_mode $reject_select_mode \
                    --strict_check True \
                    --sigma_ratio $sigma_ratio \
                    --center_q $center_q
    } 2>&1 | tee "$python_log_file"
done