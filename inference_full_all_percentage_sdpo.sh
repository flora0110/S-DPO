# The number of processes can only be one for inference
gpu1=$1;
reject_sort_metric=$2
epoch=$3
export PYTHONNOUSERSITE=1
base_model=meta-llama/Llama-3.2-1B-Instruct
sigma=0.15
q_center=0.25
for reject_select_mode in "bell_random"
do      
        CUDA_VISIBLE_DEVICES=$gpu1 python inference.py \
                --dataset lastfm \
                --external_prompt_path ./prompt/music.txt \
                --batch_size 32 \
                --base_model ${base_model} \
                --resume_from_checkpoint ./output/sdpo_neg3/${reject_sort_metric}_${reject_select_mode}_p${q_center}_sigma${sigma}_lastfm_321B_200seq_${epoch}epoch \
                >  ./output/sdpo_neg3/${reject_sort_metric}_${reject_select_mode}_p${q_center}_sigma${sigma}_lastfm_321B_200seq_${epoch}epoch/eval.log

        # CUDA_VISIBLE_DEVICES=$gpu1 python inference.py \
        #         --dataset lastfm \
        #         --external_prompt_path ./prompt/music.txt \
        #         --batch_size 32 \
        #         --base_model ${base_model} \
        #         --resume_from_checkpoint ./output/sdpo_neg3/sequence_logprob_margin_tail_random_t0.25_lastfm_321B_200seq_1epoch \
        #         >  ./output/sdpo_neg3/sequence_logprob_margin_tail_random_t0.15_lastfm_321B_200seq_1epoch/eval.log
done