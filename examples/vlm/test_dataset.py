import os
import sys
import json
import re
from PIL import Image

# --- 辅助函数 ---
def _normalize_text(text: str) -> str:
    return "".join(str(text).lower().split())

def _extract_answer_from_box(text: str) -> str:
    text_wo_think = re.sub(r"<\|think\|>.*?</\|think\|>", "", text, flags=re.DOTALL)
    text_wo_think = re.sub(r"<think>.*?</think>", "", text_wo_think, flags=re.DOTALL | re.IGNORECASE)

    m = re.search(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text_wo_think)
    if m:
        return m.group(1).strip()
    m = re.search(r"<\|answer\|>(.*?)</\|answer\|>", text_wo_think, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"<answer>(.*?)</answer>", text_wo_think, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    no_tags = re.sub(r"</?[^>]+>", "", text_wo_think)
    return no_tags.strip()

# --- Dataset 类（仅数据加载 + 预处理）---
class LazyVLMJsonlDataset:
    def __init__(self, data_path, processor, max_length=2048, base_image_path=None):
        self.data_path = data_path
        self.processor = processor
        self.max_length = max_length
        self.base_dir = os.path.dirname(data_path)
        self.base_image_path = base_image_path
        self.offsets = [0]

        print(f"Indexing dataset: {data_path}")
        with open(data_path, "rb") as f:
            while f.readline():
                self.offsets.append(f.tell())
        self.offsets.pop()
        print(f"Total samples: {len(self.offsets)}")

    def _get_image_path(self, sample):
        image_rel_path = sample.get("image") or sample.get("tos_key")
        if not image_rel_path:
            return None
        if isinstance(image_rel_path, list):
            image_rel_path = image_rel_path[0]

        if self.base_image_path:
            path = os.path.join(self.base_image_path, image_rel_path)
            if os.path.exists(path):
                return path
            else:
                print(f"Warning: Image not found at {path}")
                return None
        else:
            if not os.path.isabs(image_rel_path):
                path = os.path.join(self.base_dir, image_rel_path)
                if os.path.exists(path):
                    return path
            elif os.path.exists(image_rel_path):
                return image_rel_path
        return None

    def _build_messages(self, sample):
        user_content = ""
        user_append = ""
        if "messages" in sample:
            for msg in sample["messages"]:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    user_append = msg.get("append", "")
                    break

        messages = []
        if isinstance(user_content, str):
            if "<image>" in user_content:
                parts = user_content.split("<image>")
                content = []
                for i, part in enumerate(parts):
                    if part:
                        content.append({"type": "text", "text": part})
                    if i < len(parts) - 1:
                        content.append({"type": "image"})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": [{"type": "text", "text": user_content}]})
        elif isinstance(user_content, list):
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": str(user_content)}]})

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                base_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                print(f"apply_chat_template failed: {e}, fallback")
                base_prompt = "".join(p["text"] for p in messages[0]["content"] if p["type"] == "text")
        else:
            base_prompt = "".join(p["text"] for p in messages[0]["content"] if p["type"] == "text")

        return base_prompt + user_append

    def _extract_ground_truth(self, sample):
        raw = sample.get("gt_answer", "")
        if not raw and "messages" in sample:
            for turn in sample["messages"]:
                if turn.get("from") in ("gpt", "assistant"):
                    raw = str(turn.get("value", ""))
                    break
        return raw

    def __getitem__(self, index):
        with open(self.data_path, "r", encoding="utf-8") as f:
            f.seek(self.offsets[index])
            line = f.readline()
            sample = json.loads(line)

        image_path = self._get_image_path(sample)
        messages_text = self._build_messages(sample)
        raw_answer = self._extract_ground_truth(sample)
        clean_answer = _extract_answer_from_box(raw_answer) or raw_answer.strip()

        image_obj = None
        if image_path:
            try:
                image_obj = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Failed to load image {image_path}: {e}")

        return {
            "messages": messages_text,
            "images": image_obj,
            "answer": raw_answer,
            "clean_answer": clean_answer,
            "query_id": sample.get("query_id", sample.get("id", str(index)))
        }

# --- 主程序：加载模型 + 打印数据 ---
def main():
    if len(sys.argv) < 3:
        print("Usage: python inspect_data.py <tokenizer_path> <data_path> [image_dir]")
        sys.exit(1)

    tokenizer_path = sys.argv[1]
    data_path = sys.argv[2]
    image_dir = sys.argv[3] if len(sys.argv) > 3 else "/rice_vl/instruct/images"

    # ===== 加载模型（processor + tokenizer）=====
    print(f"Loading processor and tokenizer from: {tokenizer_path}")
    from areal.utils.hf_utils import load_hf_processor_and_tokenizer
    processor, tokenizer = load_hf_processor_and_tokenizer(tokenizer_path)

    # 添加自定义 special tokens（与训练一致）
    CUSTOM_SPECIAL_TOKENS = [
        "<think_on>", "</think_on>",
        "<think_off>", "</think_off>",
        "<answer>", "</answer>"
    ]
    tokenizer.add_tokens(CUSTOM_SPECIAL_TOKENS, special_tokens=True)
    print("Added custom special tokens:", CUSTOM_SPECIAL_TOKENS)

    # ===== 构建数据集 =====
    dataset = LazyVLMJsonlDataset(
        data_path=data_path,
        processor=processor,
        base_image_path=image_dir
    )

    # ===== 打印第一个样本 =====
    print("\n" + "="*60)
    print("Processing first sample...")
    item = dataset[0]

    print("\n--- Formatted prompt (input to model) ---")
    print(repr(item["messages"]))

    print("\n--- Raw ground truth answer (from dataset) ---")
    print(repr(item["answer"]))

    print("\n--- Clean answer (used in reward function) ---")
    print(repr(item["clean_answer"]))

    print(f"\n--- Query ID ---")
    print(item["query_id"])

    print(f"\n--- Image loaded? {'✅ Yes' if item['images'] is not None else '❌ No'}")

    print("\n✅ Data inspection completed.")

if __name__ == "__main__":
    main()

# python test_data.py /vlm/pretrain_models/Qwen2.5-VL-3B-Instruct /rice_vl/gar/math/math_filtered_classified_6_sampled_0.2k_stage1.jsonl [/your/image/root]