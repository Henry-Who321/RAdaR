import os
from datasets import load_from_disk, load_dataset, DatasetDict



def get_gsm8k_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    # dataset = load_dataset(path=path, split=split)
    dataset = load_dataset("openai/gsm8k", "main")

    def process(sample):
        print(f"[DEBUG] Available keys: {list(sample.keys())}")
        # 如果看到的是 'input' 或其他，就用那个
        # return {"content": sample["question"]}  # 或 sample["input"] 等
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_gsm8k_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):  

    if os.path.exists(path):
        print(f"[INFO] 从本地路径加载数据: {path}")
        try:
            # 尝试作为 Arrow 数据集加载 (你现在的格式)
            dataset = load_from_disk(path)
        except Exception:
            # 如果失败，尝试作为 JSON/Parquet 加载
            dataset = load_dataset("json", data_files=path, split=split)
    else:
        # 如果路径不存在，假设是 HF Hub 的 ID (如 openai/gsm8k)
        print(f"[INFO] 路径不存在，尝试从 HF Hub 加载: {path}")
        dataset = load_dataset(path, "main")

    def process(sample):
        # print(f"[DEBUG] Available keys: {list(sample.keys())}")
        messages = [
            {
                "role": "user",
                "content": sample["question"]
                + "\nPlease put your final answer within \\boxed{}.",
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["question"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:
        print(f"[INFO] 正在过滤长度超过 {max_length} 的样本...")
        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)
        print(f"[INFO] 过滤完成。保留样本数: {len(dataset)}")

    return dataset
