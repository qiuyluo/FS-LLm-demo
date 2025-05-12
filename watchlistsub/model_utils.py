import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModelForCausalLM
from huggingface_hub import snapshot_download
import os
import ast

HF_TOKEN = "hf_xxxx" # replace to actual hugging face token
REPO_ID = "Annie0430/test_fileIO"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_loaded = {}

def extract_list_from_reply(reply: str) -> list:
    """
    Extracts the first Python list literal from a string.

    Args:
        reply (str): The raw string response from the LLM.

    Returns:
        list: The extracted list if successful, otherwise an empty list.
    """
    try:
        start = reply.find('[')
        end = reply.rfind(']')
        if start != -1 and end != -1 and end > start:
            list_str = reply[start:end+1]
            result = ast.literal_eval(list_str)
            if isinstance(result, list):
                return result
    except Exception as e:
        print(f"extract_list_from_reply error: {e}")
    return []

def recommend_next_nodes(model_name: str, node: str, topk: int, textinfo: str) -> list:
    """
    Recommends the next workflow nodes given a current node and context using an LLM with LoRA adapters.

    Args:
        model_name (str): The identifier for the LoRA adapter to be loaded.
        node (str): The current workflow node.
        topk (int): Number of recommended next nodes to return.
        textinfo (str): Additional background or context information to include in the prompt.

    Returns:
        list: A list of suggested next nodes, extracted from the LLM output.
    """
    global _loaded
    key = (BASE_MODEL, model_name)
    if key in _loaded:
        model, tokenizer, device = _loaded[key]
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

        cache_dir = f"./cache/{model_name}"
        repo_dir = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
            allow_patterns=[f"test_whole_process/server/ckpt/{model_name}/*"]
        )
        adapter_dir = os.path.join(repo_dir, f"test_whole_process/server/ckpt/{model_name}")

        model = PeftModelForCausalLM.from_pretrained(
            base_model,
            adapter_dir,
            torch_dtype=torch.float32
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        _loaded[key] = (model, tokenizer, device)

    prompt = (
        f"You are a scientific workflow recommender.\n\n"
        f"Background Info:\n{textinfo}\n\n"
        f"Given the current node: '{node}', "
        f"please recommend the next {topk} workflow nodes.\n"
        f"Respond with ONLY a Python list of the next {topk} step names."
    )

    tokenizer = _loaded[key][1]
    device = _loaded[key][2]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = _loaded[key][0].generate(**inputs, max_new_tokens=64)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("LLM raw reply:", reply)

    suggestions = extract_list_from_reply(reply)
    print("Parsed suggestions:", suggestions)
    return suggestions

def get_answer(node: str, topk: int, textinfo: str) -> list:
    """
    Entry point for recommending workflow steps based on a node and context.

    Args:
        node (str): The current workflow node.
        topk (int): The number of recommendations to generate.
        textinfo (str): Background context for the model prompt.

    Returns:
        list: The recommended next workflow nodes.
    """
    print(f"In get_answer, Node: {node}, Top-K: {topk}, TextInfo: {textinfo}")
    model = "server_round_2_model_0"
    suggestions = recommend_next_nodes(model, node, topk, textinfo)
    print("recommend_next_nodes returned:", suggestions)
    return suggestions
