import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===== 基本設定 =====
method = "sequence_logprob_margin"  # "chosen", "rejected", "margins", "loss", "last_hidden_embedding_inner_prod"
category = "lastfm"

# 要畫的 p
p_list = [0, 25, 50, 75, 100]

base = "output"

# metric 設定
metric_ = "chosen"   # chosen / rejected / margins
# metric = metric_
metric = f"rewards/{metric_}"   # e.g. logps/chosen

plt.figure(figsize=(10, 6))

# ===== loop 每個 p =====
for p in p_list:
    if p == 0:
        p_str = "lowest"
    elif p == 100:
        p_str = "highest"
    else:
        p_str = f"q{p}"
    checkpoint_dir = f"./{base}/{p_str}_{method}_{category}_dpo_321B_200seq_1epoch/checkpoint-68"
    filename = f"{checkpoint_dir}/trainer_state.json"

    if not os.path.exists(filename):
        print(f"❌ Missing: p{p}")
        continue

    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        log_history = data.get("log_history", data)
        # print(log_history[0].get(metric, "N/A"))

        eval_data = []

        for entry in log_history:

            if log_history[0].get(metric, "N/A")!="N/A" and metric in entry:
                eval_data.append({
                    "step": entry["step"],
                    "value": entry[metric]
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
plt.title(f"{category} using {method} sampling: {metric} across p values")
plt.xlabel("Step")
plt.ylabel(metric)
plt.legend()
plt.grid(True)

# ===== 存圖 =====
save_dir = f"./{base}/summary_results/{method}_{category}_321B_200seq_1epoch_p_compare/"
os.makedirs(save_dir, exist_ok=True)

save_path = f"./{base}/summary_results/{method}_{category}_321B_200seq_1epoch_p_compare/{metric_}.png"
plt.savefig(save_path)
plt.show()

print(f"✅ Saved to {save_path}")