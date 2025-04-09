import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModelForCausalLM
from huggingface_hub import hf_hub_download

HF_TOKEN = "hf_xxxxxxxx"
REPO_ID = "Annie0430/test_fileIO"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_loaded = {}

def get_answer(client_id, model_name, question):
    global _loaded

    key = (BASE_MODEL, model_name)
    if key in _loaded:
        model, tokenizer, device = _loaded[key]
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

        adapter_dir = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            subfolder=f"test_whole_process/server/ckpt/{model_name}",
            token=HF_TOKEN,
            local_dir=f"./cache/{model_name}",
            local_dir_use_symlinks=False
        )

        model = PeftModelForCausalLM.from_pretrained(
            base_model,
            adapter_dir,
            torch_dtype=torch.float32
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        _loaded[key] = (model, tokenizer, device)

    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply
