# The number of processes can only be one for inference
gpu1=$1;
reject_sort_metric=$2
export PYTHONNOUSERSITE=1
base_model=meta-llama/Llama-3.2-1B-Instruct

for reject_select_mode in "lowest" "q25" "middle" "q75" "highest"
do      
        CUDA_VISIBLE_DEVICES=$gpu1 python inference.py \
                --dataset lastfm \
                --external_prompt_path ./prompt/music.txt \
                --batch_size 32 \
                --base_model ${base_model} \
                --resume_from_checkpoint ./output/${reject_sort_metric}_${reject_select_mode}_lastfm_dpo_321B_200seq_1epoch \
                >  ./output/${reject_sort_metric}_${reject_select_mode}_lastfm_dpo_321B_200seq_1epoch/eval.log
done