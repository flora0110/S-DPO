import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
# ===== 基本設定 =====
method = "sequence_logprob_margin"  # "chosen", "rejected", "margins", "loss", "last_hidden_embedding_inner_prod"
category = "lastfm"
beta=1.0
epoch = 1
if epoch == 1:
    checkpoint_step = 68   # 這是 inference 的 checkpoint，通常是最後一個 checkpoint
elif epoch == 3:
    checkpoint_step = 204   # 這是 inference 的 checkpoint，通常是最後一個 checkpoint
# 要畫的 p
# p_list = ["random", "bell_random", "bell_random_sigma0.15", 50]#, "bell_q25_random_sigma0.25", "bell_q75_random_sigma0.25"]   # 這裡的 p 是指 reject 的百分比，random 是隨機 reject
p_list = [0, 25, 50, 75, 100]

base = "output/sdpo_neg3"   # 根目錄，裡面有不同 p 的資料夾

# metric 設定
metric_num = "3"   # chosen / rejected / margins
# metric = metric_
# metric = f"rewards/{metric_}"   # e.g. logps/chosen

plt.figure(figsize=(10, 6))

# ===== loop 每個 p =====
for p in p_list:
    if base == "output/sdpo_neg3":
        if p == "random":
            checkpoint_dir = f"./{base}/random_{category}_321B_200seq_{epoch}epoch/checkpoint-{checkpoint_step}"
            filename = f"{checkpoint_dir}/trainer_state.json"
            if not os.path.exists(filename):
                print(f"❌ Missing: random")
        else:
            if p == 0:
                p_str = "lowest"
            elif p == 100:
                p_str = "highest"
            elif p==50:
                p_str = "middle"
            elif p=="bell_random":
                p_str = "bell_random"
            elif p=="bell_random_sigma0.15":
                p_str = "bell_random_sigma0.15"
            elif p=="bell_q25_random_sigma0.25":
                p_str = "bell_q25_random_sigma0.25"
            elif p=="bell_q75_random_sigma0.25":
                p_str = "bell_q75_random_sigma0.25"
            else:
                p_str = f"q{p}"
            checkpoint_dir = f"./{base}/{method}_{p_str}_{category}_321B_200seq_{epoch}epoch/checkpoint-{checkpoint_step}"
            filename = f"{checkpoint_dir}/trainer_state.json"
    else:
        if p == "random":
            checkpoint_dir = f"./{base}/{category}_dpo_321B_200seq_{epoch}epoch/checkpoint-{checkpoint_step}"
            filename = f"{checkpoint_dir}/trainer_state.json"
            if not os.path.exists(filename):
                print(f"❌ Missing: random")
        else:
            if p == 0:
                p_str = "lowest"
            elif p == 100:
                p_str = "highest"
            elif p==50:
                p_str = "middle"
            elif p=="bell_random":
                p_str = "bell_random"
            elif p=="bell_random_sigma0.15":
                p_str = "bell_random_sigma0.15"
            elif p=="bell_q25_random_sigma0.25":
                p_str = "bell_q25_random_sigma0.25"
            elif p=="bell_q75_random_sigma0.25":
                p_str = "bell_q75_random_sigma0.25"
            else:
                p_str = f"q{p}"
            checkpoint_dir = f"./{base}/{p_str}_{method}_{category}_dpo_321B_200seq_{epoch}epoch/checkpoint-{checkpoint_step}"
            filename = f"{checkpoint_dir}/trainer_state.json"
        

    if not os.path.exists(filename):
        print(f"Processing p{p} from {filename}...")
        print(f"❌ Missing: p{p}")
        continue

    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        log_history = data.get("log_history", data)
        # print(log_history[0].get(metric, "N/A"))

        eval_data = []

        for entry in log_history:
            metric_chosen = "rewards/chosen"
            metric_rejected = f"rewards/rejected{metric_num}"
            

            if log_history[0].get(metric_chosen, "N/A")!="N/A" and metric_chosen in entry and log_history[0].get(metric_rejected, "N/A")!="N/A" and metric_rejected in entry:
                # if entry["step"] % 10 == 0:
                # x2_over_x1 = math.exp(
                #     (entry[metric_rejected] - entry[metric_chosen]) / beta
                # )
                x2 = math.exp(
                    (entry[metric_rejected])
                )
                eval_data.append({
                    "step": entry["step"],
                    "value": x2
                })

        if len(eval_data) == 0:
            print(f"⚠️ No eval data for p{p}")
            continue

        df_eval = pd.DataFrame(eval_data)

        # 畫線
        plt.plot(
            df_eval["step"],
            df_eval["value"],
            marker='o',
            label=f"p{p}"
        )

    except Exception as e:
        print(f"❌ Error in p{p}: {e}")

# ===== 圖設定 =====
# plt.title(f"{category} using {method} sampling: x1x2{metric_num} across p values")
plt.title(f"{category} using {method} sampling: x2 across p values")
plt.xlabel("Step")
# plt.ylabel(f"x2_over_x1-{metric_num}")
plt.ylabel(f"x2")
plt.legend()
plt.grid(True)

# ===== 存圖 =====
save_dir = f"./{base}/summary_results/{method}_{category}_321B_200seq_{epoch}epoch_p_compare/"
os.makedirs(save_dir, exist_ok=True)

save_path = f"./{base}/summary_results/{method}_{category}_321B_200seq_{epoch}epoch_p_compare/x2_p.png"
plt.savefig(save_path)
plt.show()

print(f"✅ Saved to {save_path}")