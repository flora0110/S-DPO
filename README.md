before dpo
0.11302605210420842 0.5394789579158317

random
0.14188376753507015 0.5438877755511022

highest_avg_token_logprob_margin_lastfm_dpo_321B_200seq_1epoch
0.13346693386773548 0.5434869739478958

lowest_avg_token_logprob_margin_lastfm_dpo_321B_200seq_1epoch
0.13907815631262524 0.5214428857715431

highest_sequence_logprob_margin_lastfm_dpo_321B_200seq_1epoch
0.12785571142284569 0.5374749498997996

lowest_sequence_logprob_margin_lastfm_dpo_321B_200seq_1epoch
0.08376753507014029 0.4557114228456914

![lastfm_321B_200seq_1epoch_p_compare_loss](./output/summary_results/lastfm_321B_200seq_1epoch_p_compare/loss.png)
- 可能原因
    - loss越低 HR越低 (合理)
        - highest_avg_token_logprob_margin < lowest_avg_token_logprob_margin
        - highest_sequence_logprob_margin > lowest_sequence_logprob_margin
        - random > lowest_avg_token_logprob_margin ~ highest_avg_token_logprob_margin > lowest_sequence_logprob_margin ~ highest_sequence_logprob_margin

![lastfm_321B_200seq_1epoch_p_compare_chosen](./output/summary_results/lastfm_321B_200seq_1epoch_p_compare/chosen.png)
- 可能原因
    - 越偏離0 HR越低 (不合理)
        - highest_avg_token_logprob_margin < lowest_avg_token_logprob_margin
            - lowest_avg_token_logprob_margin 偏離較遠 (~ +1.0)
            - highest_avg_token_logprob_margin 偏離較少 (~ +0.25)
        - highest_sequence_logprob_margin > lowest_sequence_logprob_margin
            - lowest_sequence_logprob_margin 偏離較遠 (~ +1.75)
            - highest_sequence_logprob_margin 偏離較少 (~ -1.25)

![lastfm_321B_200seq_1epoch_p_compare_reject1](./output/summary_results/lastfm_321B_200seq_1epoch_p_compare/rejeckecd1.png)

# 重構
dpo neg1
0.27214428857715434 0.5386773547094188