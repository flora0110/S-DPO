import os
import json
import math
from typing import Dict, List, Any, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from accelerate import Accelerator
import fire

from Prompt import Prompt


# =========================
# 1. Utility functions
# =========================

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().item()
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
    return float(x)


# =========================
# 2. Prompt construction
# =========================

def convert_dict_to_prompt(data_point: Dict[str, Any], prompt_path: str) -> str:
    """
    Build prompt using your existing Prompt class.

    Important:
    This follows the same logic as your DPO / SDPO / inference code:
        t = Prompt(prompt_path)
        t.historyList = ...
        t.itemList = ...
        t.trueSelection = ...
        prompt = str(t)
    """
    t = Prompt(prompt_path)

    history = data_point["historyList"]
    if isinstance(history, str):
        history = history.split("::")

    t.historyList = history
    t.itemList = data_point["itemList"]
    t.trueSelection = data_point["trueSelection"]

    return str(t)


# =========================
# 3. Model loading
# =========================

def load_model_and_tokenizer(
    base_model: str,
    resume_from_checkpoint: str = "",
    load_in_4bit: bool = True,
    bf16: bool = True,
):
    """
    Load base model and optional PEFT adapter.
    """

    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
    else:
        bnb_config = None

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype if not load_in_4bit else None,
    )

    if resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)

    model.eval()

    if "Llama-3" in base_model or "Llama-3" in base_model.lower():
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    device = next(model.parameters()).device

    return model, tokenizer, device


# =========================
# 4. Core logprob scorer
# =========================

class CandidateScorer:
    """
    Main scorer class.

    Current metrics:
        - sequence_logprob
        - avg_token_logprob
        - token_len

    You can extend this class later to add:
        - CHES
        - exposure score
        - cluster membership
        - embedding distance
        - popularity bucket
        - rank margin
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        add_space_before_item: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.add_space_before_item = add_space_before_item

    def tokenize_prompt_and_item(
        self,
        prompt: str,
        item: str,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Return:
            prompt_ids
            item_ids
            full_input_ids

        We tokenize prompt and item separately to preserve the boundary.
        This is usually safer than tokenizer(prompt + item).
        """
        item_text = " " + item if self.add_space_before_item else item

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
        ).input_ids

        item_ids = self.tokenizer(
            item_text,
            add_special_tokens=False,
        ).input_ids

        input_ids = prompt_ids + item_ids

        return prompt_ids, item_ids, input_ids

    # ==============
    # Combined metric computation (remove compute_sequence_logprob())
    # ==============
    @torch.no_grad()
    def compute_candidate_metrics(
        self,
        prompt: str,
        item: str,
    ) -> Dict[str, Any]:
        """
        Compute both:
            - logprob metrics for one item
            - hidden summary for one item

        This avoids two forward passes.
        """

        prompt_ids, item_ids, input_ids = self.tokenize_prompt_and_item(
            prompt=prompt,
            item=item,
        )

        query_len = len(prompt_ids)
        item_token_len = len(item_ids)

        if item_token_len == 0:
            raise ValueError("Empty item tokenization.")

        ids_tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        outputs = self.model(
            input_ids=ids_tensor,
            output_hidden_states=True,
        )

        # =====================
        # 1. Logprob metrics
        # =====================
        logits = outputs.logits[:, :-1, :]
        labels = ids_tensor[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)

        token_log_probs = log_probs.gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)

        response_start = max(query_len - 1, 0)
        response_end = response_start + item_token_len

        response_token_log_probs = token_log_probs[:, response_start:response_end]

        if response_token_log_probs.numel() == 0:
            sequence_logprob = 0.0
            avg_token_logprob = 0.0
        else:
            sequence_logprob = safe_float(response_token_log_probs.sum())
            avg_token_logprob = safe_float(response_token_log_probs.mean())

        # =====================
        # 2. Hidden summary
        # =====================
        hidden = outputs.hidden_states[-1][0]

        # Same slicing as your reference CHES function.
        item_hidden_embed = hidden[query_len - 1:]

        if item_hidden_embed.shape[0] == 0:
            raise ValueError("Empty item hidden embeddings after slicing.")

        item_hidden_embed = item_hidden_embed.to(torch.float32)

        return {
            "sequence_logprob": sequence_logprob,
            "avg_token_logprob": avg_token_logprob,
            "token_len": item_token_len,

            # tensor summaries, kept in memory only
            "item_hidden_sum": item_hidden_embed.sum(dim=0),
            "item_hidden_len": item_hidden_embed.shape[0],
            "item_last_hidden": item_hidden_embed[-1],
        }


    def score_candidate(
        self,
        prompt: str,
        item: str,
        data_point: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Score one candidate item.

        This is the main function to extend later.
        """
        # score = self.compute_sequence_logprob(prompt, item)
        return self.compute_candidate_metrics(prompt, item)
        # Extension point:
        # Add extra metrics here later.
        #
        # Example:
        # score["candidate_exposure"] = exposure_dict.get(item, 0)
        # score["candidate_cluster"] = cluster_dict.get(item, None)
        # score["embedding_distance_to_history"] = ...
        #
        # return score

    def score_all_items(
        self,
        prompt: str,
        item_list: List[str],
        data_point: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Score all items in itemList.
        """
        scores = []

        for item_id, item in enumerate(item_list):
            item_score = self.score_candidate(
                prompt=prompt,
                item=item,
                data_point=data_point,
            )

            row = {
                "candidate_id": item_id,
                "item": item,
                **item_score,
            }

            scores.append(row)

        return scores


# =========================
# 5. Ranking and margin logic
# =========================

def add_ranks(scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add rank_by_sequence_logprob and rank_by_avg_token_logprob.

    Higher logprob is better, so descending order.
    Rank starts from 1.
    """

    by_seq = sorted(
        scores,
        key=lambda x: x["sequence_logprob"],
        reverse=True,
    )

    for rank, row in enumerate(by_seq, start=1):
        row["rank_by_sequence_logprob"] = rank

    by_avg = sorted(
        scores,
        key=lambda x: x["avg_token_logprob"],
        reverse=True,
    )

    for rank, row in enumerate(by_avg, start=1):
        row["rank_by_avg_token_logprob"] = rank

    return scores


def build_candidate_rows_for_example(
    example_id: int,
    data_point: Dict[str, Any],
    prompt: str,
    item_scores: List[Dict[str, Any]],
    include_true_row: bool = False,
    include_prompt: bool = True,
    include_item_list: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert item-level scores into candidate-level rows.

    Recommended output:
        one row = one candidate negative item

    chosen = trueSelection
    rejected = candidate_item

    margin = chosen_logprob - rejected_logprob

    If include_true_row=False:
        only negative candidates are written.
    """

    true_selection = data_point["trueSelection"]

    true_score = None
    for s in item_scores:
        if s["item"] == true_selection:
            true_score = s
            break

    if true_score is None:
        raise ValueError(
            f"trueSelection `{true_selection}` not found in itemList for example_id={example_id}"
        )

    rows = []

    for s in item_scores:
        candidate_item = s["item"]
        is_true = candidate_item == true_selection

        if is_true and not include_true_row:
            continue

        s_w = true_score["item_hidden_sum"]
        s_l = s["item_hidden_sum"]

        t_w = true_score["item_hidden_len"]
        t_l = s["item_hidden_len"]

        last_w = true_score["item_last_hidden"]
        last_l = s["item_last_hidden"]

        ches_score = (s_w * s_l).sum() - torch.norm(s_w) ** 2

        ln_ches_score = (
            ((s_w * s_l).sum() / (t_w * t_l))
            - (torch.norm(s_w) ** 2 / (t_w ** 2))
        )

        last_hidden_embedding_inner_prod = torch.inner(last_w, last_l)

        row = {
            "example_id": example_id,
            "candidate_id": s["candidate_id"],

            "historyList": data_point["historyList"],
            "trueSelection": true_selection,

            "candidate_item": candidate_item,
            "is_true_selection": is_true,

            # DPO-friendly fields
            "chosen": true_selection,
            "rejected": candidate_item,

            # chosen / trueSelection scores
            "chosen_sequence_logprob": true_score["sequence_logprob"],
            "chosen_avg_token_logprob": true_score["avg_token_logprob"],
            "chosen_token_len": true_score["token_len"],
            "chosen_rank_by_sequence_logprob": true_score["rank_by_sequence_logprob"],
            "chosen_rank_by_avg_token_logprob": true_score["rank_by_avg_token_logprob"],

            # rejected / candidate scores
            "rejected_sequence_logprob": s["sequence_logprob"],
            "rejected_avg_token_logprob": s["avg_token_logprob"],
            "rejected_token_len": s["token_len"],
            "rejected_rank_by_sequence_logprob": s["rank_by_sequence_logprob"],
            "rejected_rank_by_avg_token_logprob": s["rank_by_avg_token_logprob"],

            # margins: chosen - rejected
            "sequence_logprob_margin": true_score["sequence_logprob"] - s["sequence_logprob"],
            "avg_token_logprob_margin": true_score["avg_token_logprob"] - s["avg_token_logprob"],

            # hidden-based metrics
            "ches_score": safe_float(ches_score),
            "ln_ches_score": safe_float(ln_ches_score),
            "last_hidden_embedding_inner_prod": safe_float(last_hidden_embedding_inner_prod),
        }

        if include_prompt:
            row["prompt"] = prompt

        if include_item_list:
            row["itemList"] = data_point["itemList"]

        # Extension point:
        # If later item_scores contain extra metrics, add margins here.
        #
        # Example:
        # if "ches" in true_score and "ches" in s:
        #     row["ches_margin"] = true_score["ches"] - s["ches"]

        rows.append(row)

    return rows


# =========================
# 6. Main scoring function
# =========================

def score_dataset(
    data_path: str = "test.json",
    split_name: str = "test",
    output_path: str = "candidate_scores.jsonl",

    base_model: str = "",
    resume_from_checkpoint: str = "",
    prompt_path: str = "prompt.txt",

    max_examples: int = -1,
    include_true_row: bool = False,
    include_prompt: bool = True,
    include_item_list: bool = True,

    add_space_before_item: bool = False,
    strip_prompt_newline: bool = False,

    load_in_4bit: bool = True,
    bf16: bool = True,
):
    """
    Score each candidate item in itemList.

    Input format:
    [
      {
        "historyList": "...",
        "itemList": [...],
        "trueSelection": "..."
      }
    ]

    Output:
        candidate-level JSONL.
    """

    if base_model == "":
        raise ValueError("Please provide --base_model")

    output_dir = os.path.dirname(output_path)
    ensure_dir(output_dir)

    if os.path.exists(output_path):
        os.remove(output_path)

    print("=" * 80)
    print("Loading model...")
    print(f"base_model: {base_model}")
    print(f"resume_from_checkpoint: {resume_from_checkpoint}")
    print("=" * 80)

    model, tokenizer, device = load_model_and_tokenizer(
        base_model=base_model,
        resume_from_checkpoint=resume_from_checkpoint,
        load_in_4bit=load_in_4bit,
        bf16=bf16,
    )

    scorer = CandidateScorer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        add_space_before_item=add_space_before_item,
    )

    print("=" * 80)
    print("Loading dataset...")
    print(f"data_path: {data_path}")
    print(f"split_name: {split_name}")
    print("=" * 80)

    data_files = {split_name: data_path}
    dataset = load_dataset("json", data_files=data_files)[split_name]

    if max_examples is not None and max_examples > 0:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    print(dataset)

    total_candidate_rows = 0

    for example_id, data_point in enumerate(dataset):
        prompt = convert_dict_to_prompt(
            data_point=data_point,
            prompt_path=prompt_path,
        )

        # Match your previous inference.py behavior only if explicitly requested.
        # For DPO / SDPO consistency, default is False.
        if strip_prompt_newline:
            prompt = prompt.rstrip("\n")

        item_list = data_point["itemList"]
        true_selection = data_point["trueSelection"]

        if true_selection not in item_list:
            print(
                f"[Warning] example_id={example_id}: trueSelection not in itemList: {true_selection}"
            )
            continue

        item_scores = scorer.score_all_items(
            prompt=prompt,
            item_list=item_list,
            data_point=data_point,
        )

        item_scores = add_ranks(item_scores)

        candidate_rows = build_candidate_rows_for_example(
            example_id=example_id,
            data_point=data_point,
            prompt=prompt,
            item_scores=item_scores,
            include_true_row=include_true_row,
            include_prompt=include_prompt,
            include_item_list=include_item_list,
        )

        for row in candidate_rows:
            append_jsonl(output_path, row)

        total_candidate_rows += len(candidate_rows)

        if (example_id + 1) % 50 == 0:
            print(
                f"Processed {example_id + 1} examples, "
                f"written {total_candidate_rows} candidate rows."
            )

    print("=" * 80)
    print("Done.")
    print(f"Output path: {output_path}")
    print(f"Total examples: {len(dataset)}")
    print(f"Total candidate rows: {total_candidate_rows}")
    print("=" * 80)


if __name__ == "__main__":
    fire.Fire(score_dataset)