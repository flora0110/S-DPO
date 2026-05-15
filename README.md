
HitRatio@1, ValidRatio

before dpo
0.23246492985971945 0.9879759519038076

random
0.2913827655310621 0.9767535070140281

highest_avg_token_logprob_margin_lastfm_dpo_321B_200seq_1epoch
0.2777555110220441 0.9819639278557114

lowest_avg_token_logprob_margin_lastfm_dpo_321B_200seq_1epoch
0.2837675350701403 0.9082164328657315

- sequence_logprob_margin
    - highest
        - 0.26973947895791583 0.9803607214428858

    - q75
        - 0.2725450901803607 0.9775551102204408

    - middle
        - 0.28617234468937874 0.9743486973947896

    - q25
        - 0.2881763527054108 0.9671342685370742

    - lowest
        - 0.16472945891783566 0.6056112224448897

    - random_except_lowest
        - 0.2889779559118236 0.9739478957915831

## loss
![lastfm_321B_200seq_1epoch_p_compare_loss](./output/summary_results/lastfm_321B_200seq_1epoch_p_compare/loss.png)
- 可能原因
    - loss越低 HR越低 (不合理)
        - random ~ highest_avg_token_logprob_margin ~ lowest_avg_token_logprob_margin
        ~ highest_sequence_logprob_margin
        \>> lowest_sequence_logprob_margin
        - highest_sequence_logprob_margin比lowest_sequence_logprob_margin更低

## accuracies
![lastfm_321B_200seq_1epoch_p_compare_accuracies](./output/summary_results/lastfm_321B_200seq_1epoch_p_compare/accuracies.png)
- 可能原因
    - accuracies越高 HR越低 (不合理)
        - random ~ highest_avg_token_logprob_margin ~ lowest_avg_token_logprob_margin
        ~ highest_sequence_logprob_margin
        \>> lowest_sequence_logprob_margin
        - highest_sequence_logprob_margin比lowest_sequence_logprob_margin更高

## chosen
![lastfm_321B_200seq_1epoch_p_compare_chosen](./output/summary_results/lastfm_321B_200seq_1epoch_p_compare/chosen.png)
- 可能原因
    - chosen reward越偏離0 HR越低 (可能合理？)-
        - random ~ highest_avg_token_logprob_margin ~ lowest_avg_token_logprob_margin
        ~ highest_sequence_logprob_margin
        \>> lowest_sequence_logprob_margin
        - |lowest_sequence_logprob_margin| > 1.5 > others
    - chosen reward越震盪 HR越低 (可能合理？)-
        - random ~ highest_avg_token_logprob_margin ~ lowest_avg_token_logprob_margin
        ~ highest_sequence_logprob_margin
        \>> lowest_sequence_logprob_margin
        - lowest_sequence_logprob_margin 振蕩幅度 > others
            - 可能調低lr?

## reject
![lastfm_321B_200seq_1epoch_p_compare_reject1](./output/summary_results/lastfm_321B_200seq_1epoch_p_compare/rejected1.png)
- 可能原因
    - reject reward越低 HR越低 (不合理)-
        - random ~ highest_avg_token_logprob_margin ~ lowest_avg_token_logprob_margin
        ~ highest_sequence_logprob_margin
        \>> lowest_sequence_logprob_margin
        - highest_sequence_logprob_margin比lowest_sequence_logprob_margin更低

## margin between chosen and reject reward
![lastfm_321B_200seq_1epoch_p_compare_reject1](./output/summary_results/lastfm_321B_200seq_1epoch_p_compare/margins-rejected1.png)
- 可能原因
    - margin between chosen and reject reward越高 HR越低 (不合理)-
        - random ~ highest_avg_token_logprob_margin ~ lowest_avg_token_logprob_margin
        ~ highest_sequence_logprob_margin
        \>> lowest_sequence_logprob_margin
        - highest_sequence_logprob_margin跟lowest_sequence_logprob_margin差不多
# 重構
before dpo
0.4961923847695391 0.9947895791583167

dpo neg1
0.5202404809619239 0.9911823647294589

sdpo neg3
0.5338677354709419 0.9915831663326653