from flask import Flask, render_template, request, redirect, url_for, jsonify
from huggingface_hub import HfApi, list_repo_files
import json
import os
import tempfile
from model_utils import get_answer

HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your actual token
REPO_ID = "Annie0430/test_fileIO"
CKPT_PREFIX = "test_whole_process/server/ckpt/"

def get_model_names_from_huggingface():
    all_files = list_repo_files(REPO_ID, repo_type="dataset", token=HF_TOKEN)
    models = set()
    for path in all_files:
        if path.startswith(CKPT_PREFIX):
            parts = path[len(CKPT_PREFIX):].split('/')
            if len(parts) >= 1 and parts[0]:
                models.add(parts[0])
    return sorted(models)
app = Flask(__name__)

@app.route('/client1')
def client1_home():
    return render_template('client1.html')

@app.route('/client2')
def client2_home():
    return render_template('client2.html')

@app.route('/client1/ask')
def client1_ask():
    model_list = get_model_names_from_huggingface()
    print(f"Model list for client1: {model_list}")
    return render_template('client1_ask.html', model_list=model_list)

@app.route('/client2/ask')
def client2_ask():
    model_list = get_model_names_from_huggingface()
    return render_template('client2_ask.html', model_list=model_list)


@app.route('/upload/<client_id>', methods=['POST'])
def upload(client_id):
    file = request.files['file']
    if file and file.filename.endswith('.json'):
        json_data = json.load(file)

        id_number = client_id.replace("client", "")

        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp_file:
            json.dump(json_data, tmp_file, indent=2)
            tmp_file_path = tmp_file.name

        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=tmp_file_path,
                path_in_repo=f"demo/client_{id_number}_raw.json",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
        finally:
            os.remove(tmp_file_path)

    return redirect(url_for(f'{client_id}_ask'))

@app.route('/ask/<client_id>', methods=['POST'])
def ask(client_id):
    model = request.form.get('model')
    question = request.form.get('question')
    answer = get_answer(client_id, model, question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
