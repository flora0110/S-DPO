# The number of processes can only be one for inference
gpu1=$1;
export PYTHONNOUSERSITE=1
CUDA_VISIBLE_DEVICES=$gpu1 python inference.py \
        --dataset lastfm \
        --external_prompt_path ./prompt/music.txt \
        --batch_size 32 \
        --base_model meta-llama/Llama-3.2-1B-Instruct \
        --resume_from_checkpoint ./output/lastfm_sdpo_321B_200seq_1epoch \
	>  ./output/lastfm_sdpo_321B_200seq_1epoch/eval.log