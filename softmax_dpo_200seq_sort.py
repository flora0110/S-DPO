import os
import torch
import re
import random

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
# from trl import DPOTrainer
from trainer.softmax_dpo_trainer import DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# from utils import find_all_linear_names, print_trainable_parameters
from transformers import LlamaForCausalLM, LlamaTokenizer

from Prompt import Prompt

import torch
import bitsandbytes as bnb
from accelerate import Accelerator
import fire

import json
from collections import defaultdict

random.seed(1958)

def normalize_history(history):
    if isinstance(history, str):
        return history.split("::")
    return history


def load_candidate_scores(candidate_score_path):
    """
    Read candidate-level JSONL and build:
        score_by_example[example_id] = [candidate rows...]
    """
    score_by_example = defaultdict(list)

    with open(candidate_score_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            example_id = int(row["example_id"])
            score_by_example[example_id].append(row)

    return dict(score_by_example)

def check_candidate_rows_match_example(
    candidate_rows,
    example_id,
    true_selection,
    item_list,
    history_list,
    strict_check=True,
):
    """
    Check whether candidate rows from score file match current raw example.
    """

    if len(candidate_rows) == 0:
        raise ValueError(f"No candidate rows found for example_id={example_id}")

    first_row = candidate_rows[0]

    # 1. Check trueSelection
    if first_row.get("trueSelection") != true_selection:
        msg = (
            f"[Mismatch] example_id={example_id}: trueSelection mismatch. "
            f"score_file={first_row.get('trueSelection')} vs raw_example={true_selection}"
        )
        if strict_check:
            raise ValueError(msg)
        else:
            print("[Warning]", msg)

    # 2. Check historyList
    score_history = normalize_history(first_row.get("historyList"))
    raw_history = normalize_history(history_list)

    if score_history != raw_history:
        msg = (
            f"[Mismatch] example_id={example_id}: historyList mismatch.\n"
            f"score_file={score_history}\n"
            f"raw_example={raw_history}"
        )
        if strict_check:
            raise ValueError(msg)
        else:
            print("[Warning]", msg)

    # 3. Check itemList
    score_item_list = first_row.get("itemList", None)

    if score_item_list is not None:
        if score_item_list != item_list:
            msg = (
                f"[Mismatch] example_id={example_id}: itemList mismatch.\n"
                f"score_file={score_item_list}\n"
                f"raw_example={item_list}"
            )
            if strict_check:
                raise ValueError(msg)
            else:
                print("[Warning]", msg)

    # 4. Check every candidate
    item_set = set(item_list)

    for row in candidate_rows:
        candidate = row.get("candidate_item", row.get("rejected"))

        if candidate not in item_set:
            msg = (
                f"[Mismatch] example_id={example_id}: candidate `{candidate}` "
                f"not in current itemList."
            )
            if strict_check:
                raise ValueError(msg)
            else:
                print("[Warning]", msg)

        if candidate == true_selection:
            msg = (
                f"[Mismatch] example_id={example_id}: candidate `{candidate}` "
                f"is equal to trueSelection."
            )
            if strict_check:
                raise ValueError(msg)
            else:
                print("[Warning]", msg)

def select_around_quantile(sorted_rows, neg_num, q):
    """
    Select neg_num rows around a quantile position.

    Args:
        sorted_rows:
            rows sorted by sort_metric in ascending order.

        neg_num:
            number of negatives to select.

        q:
            quantile position.
            q=0.25 -> around 25th percentile
            q=0.50 -> around middle
            q=0.75 -> around 75th percentile
    """
    n = len(sorted_rows)

    if n < neg_num:
        raise ValueError(
            f"Not enough rows to select around quantile. "
            f"num_rows={n}, neg_num={neg_num}"
        )

    center_idx = int(round(q * (n - 1)))

    half = neg_num // 2
    start = center_idx - half
    end = start + neg_num

    # Shift window if it goes out of boundary.
    if start < 0:
        start = 0
        end = neg_num

    if end > n:
        end = n
        start = n - neg_num

    return sorted_rows[start:end]

def select_rejected_items_from_scores(
    score_by_example,
    example_id,
    true_selection,
    item_list,
    history_list,
    neg_num,
    sort_metric="avg_token_logprob_margin",
    select_mode="lowest",
    strict_check=True,
):
    """
    Select rejected items from precomputed candidate scores.

    Args:
        score_by_example:
            dict, example_id -> candidate rows

        example_id:
            original dataset index

        true_selection, item_list, history_list:
            raw example fields, used for consistency checking

        neg_num:
            number of rejected items to select

        sort_metric:
            metric used for sorting.
            Examples:
                - avg_token_logprob_margin
                - sequence_logprob_margin
                - rejected_avg_token_logprob
                - rejected_sequence_logprob

        select_mode:
            lowest:
                select smallest metric values.
                If sort_metric = avg_token_logprob_margin,
                this means hardest negatives.

            highest:
                select largest metric values.
                If sort_metric = avg_token_logprob_margin,
                this means easiest negatives.

            both:
                select some from lowest and some from highest.

    Returns:
        rejected_items: List[str]
    """

    if example_id not in score_by_example:
        raise ValueError(f"example_id={example_id} not found in candidate score file.")

    candidate_rows = score_by_example[example_id]

    check_candidate_rows_match_example(
        candidate_rows=candidate_rows,
        example_id=example_id,
        true_selection=true_selection,
        item_list=item_list,
        history_list=history_list,
        strict_check=strict_check,
    )

    # Only keep valid negative candidates.
    valid_rows = []
    for row in candidate_rows:
        candidate = row.get("candidate_item", row.get("rejected"))

        if candidate == true_selection:
            continue

        if candidate not in item_list:
            continue

        if sort_metric not in row:
            raise ValueError(
                f"sort_metric `{sort_metric}` not found in candidate row. "
                f"Available keys: {list(row.keys())}"
            )

        valid_rows.append(row)

    if len(valid_rows) < neg_num:
        raise ValueError(
            f"example_id={example_id} has only {len(valid_rows)} valid negatives, "
            f"but neg_num={neg_num}."
        )

    # Remove duplicate candidate items if any.
    seen = set()
    dedup_rows = []
    for row in valid_rows:
        candidate = row.get("candidate_item", row.get("rejected"))
        if candidate not in seen:
            seen.add(candidate)
            dedup_rows.append(row)

    valid_rows = dedup_rows

    if len(valid_rows) < neg_num:
        raise ValueError(
            f"example_id={example_id} has only {len(valid_rows)} unique valid negatives, "
            f"but neg_num={neg_num}."
        )

    ascending_rows = sorted(
        valid_rows,
        key=lambda x: x[sort_metric],
        reverse=False,
    )

    descending_rows = sorted(
        valid_rows,
        key=lambda x: x[sort_metric],
        reverse=True,
    )

    if select_mode == "lowest":
        selected_rows = ascending_rows[:neg_num]

    elif select_mode == "highest":
        selected_rows = descending_rows[:neg_num]

    elif select_mode == "both":
        # Example:
        # neg_num=2 -> 1 lowest + 1 highest
        # neg_num=3 -> 2 lowest + 1 highest
        num_low = (neg_num + 1) // 2
        num_high = neg_num - num_low

        selected_rows = []

        for row in ascending_rows:
            if len(selected_rows) >= num_low:
                break
            selected_rows.append(row)

        selected_items = {
            row.get("candidate_item", row.get("rejected"))
            for row in selected_rows
        }

        for row in descending_rows:
            candidate = row.get("candidate_item", row.get("rejected"))
            if candidate in selected_items:
                continue

            selected_rows.append(row)
            selected_items.add(candidate)

            if len(selected_rows) >= neg_num:
                break
    elif select_mode == "middle":
        selected_rows = select_around_quantile(
            sorted_rows=ascending_rows,
            neg_num=neg_num,
            q=0.50,
        )

    elif select_mode == "q25":
        selected_rows = select_around_quantile(
            sorted_rows=ascending_rows,
            neg_num=neg_num,
            q=0.25,
        )

    elif select_mode == "q75":
        selected_rows = select_around_quantile(
            sorted_rows=ascending_rows,
            neg_num=neg_num,
            q=0.75,
        )

    else:
        raise ValueError(
            f"Unknown select_mode={select_mode}. "
            f"Use one of: lowest, highest, both."
        )

    rejected_items = [
        row.get("candidate_item", row.get("rejected"))
        for row in selected_rows
    ]

    return rejected_items

def train(
    #train
    output_dir="",
    logging_dir="",
    model_name ="",
    prompt_path = "",
    dataset="",
    resume_from_checkpoint: str = "",  # either training checkpoint or final adapter
    # wandb config
    wandb_project: str = "",
    wandb_name: str = "",   # the name of the wandb run
    # training hyperparameters
    beta: float = 0.1,
    neg_num: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-5,
    cutoff_len: int = 512,
    eval_step = 1,
    train_data_path = "",
    val_data_path = "",
    train_candidate_score_path: str = "S-DPO/toy_200seq/train_candidate_scores.jsonl",
    val_candidate_score_path: str = "S-DPO/toy_200seq/val_candidate_scores.jsonl",
    reject_sort_metric: str = "avg_token_logprob_margin",
    reject_select_mode: str = "lowest",  # lowest / highest / both
    strict_check: bool = True,
):
    
    data_files = {
        # "train": "./toy_200seq/lastfm-train.json",
        # "validation": "./toy_200seq/lastfm-val.json",
        "train": train_data_path,
        "validation": val_data_path,
    }

    train_score_by_example = load_candidate_scores(train_candidate_score_path)
    val_score_by_example = load_candidate_scores(val_candidate_score_path)




    def convert_dict_to_prompt(d:dict):
        t = Prompt(prompt_path)
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t
    
    def make_process_data(score_by_example, split_name="train"):
        def process_data(examples, indices):
            dic = {"prompt":[], "chosen":[]}

            for i in range(1, neg_num+1):
                dic[f"rejected{i}"] = []

            columns = list(examples.keys())
            for local_i in range(len(examples[columns[0]])):
                example_id = int(indices[local_i])

                data_point = {}
                data_point["trueSelection"] = examples["trueSelection"][local_i]
                data_point["itemList"] = examples["itemList"][local_i]
                data_point["historyList"] = examples["historyList"][local_i] 

                t = convert_dict_to_prompt(data_point)
                prompt = str(t)
                chosen = data_point["trueSelection"]

                selected_rejected_items = select_rejected_items_from_scores(
                    score_by_example=score_by_example,
                    example_id=example_id,
                    true_selection=data_point["trueSelection"],
                    item_list=data_point["itemList"],
                    history_list=data_point["historyList"],
                    neg_num=neg_num,
                    sort_metric=reject_sort_metric,
                    select_mode=reject_select_mode,
                    strict_check=strict_check,
                )

                # negative_items = [item for item in data_point["itemList"] if item != data_point["trueSelection"]]
                # sample_negs = random.sample(negative_items, neg_num)
                dic["prompt"].append(prompt)
                dic["chosen"].append(chosen)
                # cnt = 0  
                # for rejected in sample_negs:
                #     cnt += 1
                #     dic[f"rejected{cnt}"].append(rejected)

                for j, rejected in enumerate(selected_rejected_items, start=1):
                    dic[f"rejected{j}"].append(rejected)
            return dic
        return process_data


    data = load_dataset("json", data_files=data_files)

    process_train_data = make_process_data(
        score_by_example=train_score_by_example,
        split_name="train",
    )

    process_val_data = make_process_data(
        score_by_example=val_score_by_example,
        split_name="validation",
    )

    columns = data["train"].column_names
    # train_data = data["train"].map(process_data, remove_columns=columns, \
    #                                 num_proc=8, batched=True).shuffle(seed=42)
    train_data = data["train"].map(process_train_data, remove_columns=columns, \
                                    num_proc=8, batched=True, with_indices=True).shuffle(seed=42)

    print(train_data)

    # random 2000 samples for validation
    # val_data = data["validation"].map(process_data, remove_columns=columns, \
    #                                     num_proc=8, batched=True).shuffle(seed=42)
    
    val_data = data["validation"].map(process_val_data, remove_columns=columns, \
                                        num_proc=8, batched=True, with_indices=True).shuffle(seed=42)
    if val_data.num_rows > 2000:
        val_data = val_data.select(range(2000))
    
    print(val_data)

    device_index = Accelerator().process_index
    device_map = {"": device_index}
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


    base_model = LlamaForCausalLM.from_pretrained(model_name, 
                                                device_map=device_map, 
                                                # load_in_8bit=True,
                                                # torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    base_model = PeftModel.from_pretrained(base_model, resume_from_checkpoint, 
                                        is_trainable=True)
    # print_trainable_parameters(base_model)
    base_model.print_trainable_parameters()

    model_ref = LlamaForCausalLM.from_pretrained(model_name,
                                                device_map=device_map, 
                                                # load_in_8bit=True,
                                                # torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)
    reference_model = PeftModel.from_pretrained(model_ref, resume_from_checkpoint)
    reference_model.print_trainable_parameters()


    # tokenizer = LlamaTokenizer.from_pretrained(model_name)
    if 'Llama-3' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=num_train_epochs, 
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=eval_step,
        save_total_limit=100,
        evaluation_strategy="steps",
        eval_steps=eval_step,
        load_best_model_at_end=True,
        logging_steps=1,
        output_dir=output_dir,
        report_to=[],
        # report_to = "wandb",
        # run_name = wandb_name,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={'use_reentrant': True}, 
        ddp_find_unused_parameters=False,
    )

    dpo_trainer = DPOTrainer(
        base_model,
        reference_model,
        args=training_args,
        beta=beta,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len,
    )


    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)


    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)