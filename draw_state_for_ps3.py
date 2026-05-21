import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===== 基本設定 =====
method = "sequence_logprob_margin"  # "chosen", "rejected", "margins", "loss", "last_hidden_embedding_inner_prod"
category = "lastfm"
epoch = 3
if epoch == 1:
    checkpoint_step = 68   # 這是 inference 的 checkpoint，通常是最後一個 checkpoint
elif epoch == 3:
    checkpoint_step = 1122   # 這是 inference 的 checkpoint，通常是最後一個 checkpoint
# 要畫的 p
p_list = ["random_dpo", "random_sdpo", "bell_random_sigma0.25"]   # 這裡的 p 是指 reject 的百分比，random 是隨機 reject

base = "output"   # 根目錄，裡面有不同 p 的資料夾

# metric 設定
metric_ = "margins-rejected3"   # chosen / rejected / margins
# metric = metric_
metric = f"rewards/{metric_}"   # e.g. logps/chosen

plt.figure(figsize=(10, 6))

# ===== loop 每個 p =====
for p in p_list:
    if p == "random_dpo":
        checkpoint_dir = f"./{base}/{category}_dpo_321B/checkpoint-{checkpoint_step}"
        filename = f"{checkpoint_dir}/trainer_state.json"
        if not os.path.exists(filename):
            print(f"❌ Missing: random")
    elif p == "random_sdpo":
        checkpoint_dir = f"./{base}/{category}_sdpo_321B/checkpoint-{checkpoint_step}"
        filename = f"{checkpoint_dir}/trainer_state.json"
        if not os.path.exists(filename):
            print(f"❌ Missing: bell_random")
    else:
        if p == "bell_random_sigma0.25":
            p_str = "bell_random_sigma0.25"
            checkpoint_dir = f"./{base}/sdpo_neg3/{method}_{p_str}_{category}_321B/checkpoint-{checkpoint_step}"
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

            if log_history[0].get(metric, "N/A")!="N/A" and metric in entry:
                if entry["step"] % 50 == 0:   # 每 100 step 畫一點
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
            label=f"{p}"
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
save_dir = f"./{base}/summary_results/{category}_321B_compare/"
os.makedirs(save_dir, exist_ok=True)

save_path = f"./{base}/summary_results/{category}_321B_compare/{metric_}.png"
plt.savefig(save_path)
plt.show()

print(f"✅ Saved to {save_path}")