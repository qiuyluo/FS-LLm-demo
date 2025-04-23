import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModelForCausalLM
from huggingface_hub import hf_hub_download
import ast

HF_TOKEN = "hf_xxXxXxXxXxXxXxXxXxXx"  # Replace with your actual token
REPO_ID = "Annie0430/test_fileIO"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_loaded = {}

# def get_answer(model_name, node):
#     """
#     Given a model name and a node name, load the model if not cached,
#     prompt the model to recommend the next workflow node.
#     The model should respond with ONLY a Python list of suggestions (e.g. ["step_a", "step_b"]).
#     This function parses the output and returns a Python list.
#     """
#     global _loaded

#     key = (BASE_MODEL, model_name)
#     if key in _loaded:
#         model, tokenizer, device = _loaded[key]
#     else:
#         # Load tokenizer and base model
#         tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
#         base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

#         # Download and load LoRA adapter
#         adapter_dir = hf_hub_download(
#             repo_id=REPO_ID,
#             repo_type="dataset",
#             subfolder=f"test_whole_process/server/ckpt/{model_name}",
#             token=HF_TOKEN,
#             local_dir=f"./cache/{model_name}",
#             local_dir_use_symlinks=False
#         )

#         model = PeftModelForCausalLM.from_pretrained(
#             base_model,
#             adapter_dir,
#             torch_dtype=torch.float32
#         )

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device).eval()
#         _loaded[key] = (model, tokenizer, device)

#     # Construct the English prompt
#     prompt = (
#         f"You are a scientific workflow recommender. Given the current node: '{node}', "
#         "please recommend the next workflow node. Respond with a Python list of suggestions only. Do not include any extra explanation or information."
#     )

#     # Tokenize and send to model
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=256)
#     reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Parse the output as a Python list
#     try:
#         # Try extracting the first list-like structure from the reply
#         start = reply.find('[')
#         end = reply.find(']', start)
#         if start != -1 and end != -1:
#             list_str = reply[start:end+1]
#             result = ast.literal_eval(list_str)
#             if isinstance(result, list):
#                 return result
#         # Fallback: try to parse the whole reply
#         result = ast.literal_eval(reply)
#         if isinstance(result, list):
#             return result
#         # If parsing fails, return an empty list
#         return []
#     except Exception:
#         # Fallback: return empty list if parsing fails
#         return []


def get_answer(model, node):
    print("In get_answer")
    next_steps = ["step_a", "step_b", "step_c"]
    return next_steps