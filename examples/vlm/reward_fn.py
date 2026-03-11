import re
from .utils import (
    _normalize_text,
    extract_answer_fallback,
    get_mode,
    check_answer_match,
    calculate_length_penalty,
    stage2_process_completions_for_reward,
    stage1_2_process_completions_for_reward,
    stage1_1_process_completions_for_reward
)

def RAdar_stage1_2_reward_fn(prompt, completions, answer, **kwargs):
    """
    Reward function for Radar Stage 1.2 training (prompt-based mode detection).
    
    Ported from radar_stage1_2_train.py's step2_format_and_accuracy_reward_fn.
    """
    mode, _, completions_text = stage1_2_process_completions_for_reward(completions, prompt)

    # --- Constants & Configuration ---
    R_THINK_TAG = 0.2            # Reward for correct thinking tag pairs
    R_ANSWER_TAG = 0.05          # Reward for correct answer tag pairs
    R_CORRECT = 0.75             # Reward for correct answer content
    THINK_PENALTY_LIMIT = 0.1    # Maximum penalty for thinking content when thinking should be disabled
    ORDER_ERROR_PENALTY = 0.8    # Penalty if <answer> appears before thinking tags

    current_reward = 0.0
    format_ok = False
    answer_ok = False
    format_details = {"penalty": 0.0, "think_length": 0}

    target_think_tag = "think_on" if mode == "should_reasoning" else "think_off"
    think_pattern = rf'<{target_think_tag}>(.*?)</{target_think_tag}>'
    think_match = re.search(think_pattern, completions_text, re.DOTALL | re.IGNORECASE)

    if think_match:
        # Matched correct thinking tags
        think_content = think_match.group(1).strip()
        format_details["think_length"] = len(think_content)
        
        # If direct mode, content should be empty or penalized
        if mode == "should_direct":
            if len(think_content) > 0:
                length_penalty = calculate_length_penalty(len(think_content), R_penalty_limit=THINK_PENALTY_LIMIT)
                format_details["penalty"] = length_penalty
            else:
                format_details["penalty"] = 0.0
        else: # reasoning mode
            format_details["penalty"] = 0.0

        current_reward += R_THINK_TAG

    if format_details["penalty"] > 0.0:
        current_reward -= format_details["penalty"]

    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, completions_text, re.DOTALL | re.IGNORECASE)

    if answer_match:
        current_reward += R_ANSWER_TAG

    if think_match and answer_match:
        think_pos = completions_text.lower().find(f"<{target_think_tag}>")
        answer_pos = completions_text.lower().find("<answer>")
        if think_pos > answer_pos:
            current_reward -= ORDER_ERROR_PENALTY # Order error penalty
        else:
            format_ok = True
    
    # GT Extraction
    pred_answer_content = extract_answer_fallback(completions_text)
    gt_content = extract_answer_fallback(answer)

    pred_norm = _normalize_text(pred_answer_content)
    gt_norm = _normalize_text(gt_content)

    answer_ok = check_answer_match(pred_norm, gt_norm)

    if answer_ok:
        current_reward += R_CORRECT

    print(f"Mode: {mode}| Format:{format_ok}| Answer:{answer_ok}| Reward:{current_reward}\n{'='*60}| Penalty:{format_details['penalty']}|Think_length: {format_details['think_length']}\n")
    return current_reward

def RAdar_stage2_reward_fn(prompt, completions, answer, **kwargs):
    """
    Reward function for Radar Stage 2 training.

    This function evaluates the model's completion based on:
    1.  Format Adherence: Checks for correct usage of thinking tags (`<think_on>` or `<think_off>`) 
        and answer tags (`<answer>`).
    2.  Reasoning Mode:
        -   If `mode == "should_reasoning"`, expects `<think_on>...</think_on><answer>...</answer>`.
        -   If `mode == "should_direct"`, expects `<think_off>...</think_off><answer>...</answer>` and penalizes non-empty thinking content.
    3.  Tag Order: Ensures thinking tags appear before answer tags.
    4.  Answer Correctness: Compares the extracted answer against the ground truth.

    Args:
        prompt (str): The input prompt.
        completions (str): The model's generated completion.
        answer (str): The ground truth answer string (containing mode info and correct answer).
        **kwargs: Additional arguments.

    Returns:
        float: The calculated reward score.
    """
    
    # --- Constants & Configuration ---
    R_THINK_TAG = 0.2            # Reward for correct thinking tag pairs
    R_ANSWER_TAG = 0.05          # Reward for correct answer tag pairs
    R_CORRECT = 0.75             # Reward for correct answer content
    THINK_PENALTY_LIMIT = 0.1    # Maximum penalty for thinking content when thinking should be disabled
    ORDER_ERROR_PENALTY = 0.8    # Penalty if <answer> appears before thinking tags
    END_TOKEN = "<|im_end|>"

    # --- Preprocessing ---
    # Remove trailing end tokens from completion and add suffix
    completions_text = stage2_process_completions_for_reward(completions, END_TOKEN)

    # Determine the expected mode from the ground truth answer string
    mode = get_mode(answer) 
    
    # --- Reward Calculation ---
    current_reward = 0.0
    format_details = {"think_length": 0, "penalty": 0.0}
    
    # 1. Evaluate Thinking Tags
    target_think_tag = "think_on" if mode == "should_reasoning" else "think_off"
    # Regex to capture content between <tag> and </tag>
    think_pattern = rf'<{target_think_tag}>(.*?)</{target_think_tag}>'
    think_match = re.search(think_pattern, completions_text, re.DOTALL | re.IGNORECASE)

    if think_match:
        # Reward for finding the correct thinking tags
        current_reward += R_THINK_TAG
        
        think_content = think_match.group(1).strip()
        format_details["think_length"] = len(think_content)

        # Apply penalty if we are in "direct" mode but there is thinking content
        if mode == "should_direct" and len(think_content) > 0:
            penalty = calculate_length_penalty(len(think_content), THINK_PENALTY_LIMIT)
            format_details["penalty"] = penalty
            current_reward -= penalty

    # 2. Evaluate Answer Tags
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, completions_text, re.DOTALL | re.IGNORECASE)

    if answer_match:
        # Reward for finding the answer tags
        current_reward += R_ANSWER_TAG

    # 3. Evaluate Tag Order
    format_ok = False
    if think_match and answer_match:
        # Find positions of the opening tags
        think_pos = completions_text.lower().find(f"<{target_think_tag}>")
        answer_pos = completions_text.lower().find("<answer>")
        
        if think_pos > answer_pos:
            current_reward -= ORDER_ERROR_PENALTY
        else:
            format_ok = True

    # 4. Evaluate Answer Correctness
    # Extract answer content from both prediction and ground truth
    pred_answer_content = extract_answer_fallback(completions_text)
    gt_answer_content = extract_answer_fallback(answer)

    # Normalize texts for comparison
    pred_norm = _normalize_text(pred_answer_content)
    gt_norm = _normalize_text(gt_answer_content)

    answer_ok = check_answer_match(pred_norm, gt_norm)

    if answer_ok:
        current_reward += R_CORRECT

    # --- Logging & Return ---
    print(
        f"Mode: {mode} | Format: {format_ok} | Answer: {answer_ok} | Reward: {current_reward:.4f}\n"
        f"{'='*60}\n"
        f"Think Length: {format_details['think_length']} | Penalty: {format_details['penalty']:.4f}\n"
    )
    
    return current_reward


def RAdar_stage1_1_reward_fn(prompt, completions, answer, **kwargs):
    """
    Reward function for Radar Stage 1.1 training.

    This function evaluates the model's completion based on:
    1.  Format Adherence: Checks for correct usage of thinking tags (`<think_on>` or `<think_off>`) 
        and answer tags (`<answer>`).
    2.  Reasoning Mode:
        -   If `mode == "should_reasoning"`, expects `<think_on>...</think_on><answer>...</answer>`.
        -   If `mode == "should_direct"`, expects `<think_off>...</think_off><answer>...</answer>` and penalizes non-empty thinking content.
    3.  Tag Order: Ensures thinking tags appear before answer tags.
    4.  Answer Correctness: Compares the extracted answer against the ground truth.

    Args:
        prompt (str): The input prompt.
        completions (str): The model's generated completion.
        answer (str): The ground truth answer string (containing mode info and correct answer).
        **kwargs: Additional arguments.

    Returns:
        float: The calculated reward score.
    """
    
    # --- Constants & Configuration ---
    R_ANSWER_TAG = 0.3          # Reward for correct answer tag pairs
    R_CORRECT = 0.7             # Reward for correct answer content

    # --- Preprocessing ---
    # Remove trailing end tokens from completion and add suffix
    completions = stage1_1_process_completions_for_reward(completions)
    
    prefix = "<answer>"
    completions_text = prefix + (completions if completions else "")
    answer = prefix + (answer if answer else " ")
    
    # --- Reward Calculation ---
    current_reward = 0.0

    # 2. Evaluate Answer Tags
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, completions_text, re.DOTALL | re.IGNORECASE)

    if answer_match:
        # Reward for finding the answer tags
        current_reward += R_ANSWER_TAG

    # 4. Evaluate Answer Correctness
    # Extract answer content from both prediction and ground truth
    pred_answer_content = extract_answer_fallback(completions_text)
    gt_answer_content = extract_answer_fallback(answer)

    # Normalize texts for comparison
    pred_norm = _normalize_text(pred_answer_content)
    gt_norm = _normalize_text(gt_answer_content)

    answer_ok = check_answer_match(pred_norm, gt_norm)

    if answer_ok:
        current_reward += R_CORRECT

    # --- Logging & Return ---
    print(
        f"Answer: {answer_ok} | Reward: {current_reward:.4f}\n"
        f"{'='*60}\n"
    )
    
    return current_reward