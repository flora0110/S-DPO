export PYTHONNOUSERSITE=1
gpu1=$1;
CUDA_VISIBLE_DEVICES=$gpu1 python score_candidates_w_hidden.py \
  --data_path ./sample_data/lastfm-sft-cans20/lastfm-train.json \
  --split_name train \
  --output_path ./sample_data/lastfm-sft-cans20/train_candidate_scores_w_hidden.jsonl \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --resume_from_checkpoint ./output/lastfm_sft_321B \
  --prompt_path ./prompt/music.txt \
  --max_examples -1


CUDA_VISIBLE_DEVICES=$gpu1 python score_candidates_w_hidden.py \
  --data_path ./sample_data/lastfm-sft-cans20/lastfm-val.json \
  --split_name val \
  --output_path ./sample_data/lastfm-sft-cans20/val_candidate_scores_w_hidden.jsonl \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --resume_from_checkpoint ./output/lastfm_sft_321B \
  --prompt_path ./prompt/music.txt \
  --max_examples -1