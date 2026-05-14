export PYTHONNOUSERSITE=1
gpu1=$1;
CUDA_VISIBLE_DEVICES=$gpu1 python score_candidates.py \
  --data_path ./toy_200seq/lastfm-val.json \
  --split_name val \
  --output_path ./toy_200seq/val_candidate_scores.jsonl \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --resume_from_checkpoint ./output/lastfm_sft_321B_200seq \
  --prompt_path ./prompt/music.txt \
  --max_examples -1