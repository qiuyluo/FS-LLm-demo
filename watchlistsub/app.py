from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
from model_utils import initialize_model, get_answer

app = Flask(__name__)

@app.route('/client1')
def client1_home():
    return render_template('client1.html')

@app.route('/client2')
def client2_home():
    return render_template('client2.html')

@app.route('/client1/ask')
def client1_ask():
    return render_template('client1_ask.html', model_list=["base model", "custom model"])

@app.route('/client2/ask')
def client2_ask():
    return render_template('client2_ask.html', model_list=["base model", "custom model"])

@app.route('/upload/<client_id>', methods=['POST'])
def upload(client_id):
    file = request.files['file']
    if file and file.filename.endswith('.json'):
        content = json.load(file)
        initialize_model(client_id, content)
    return redirect(url_for(f'{client_id}_ask'))

@app.route('/ask/<client_id>', methods=['POST'])
def ask(client_id):
    model = request.form.get('model')
    question = request.form.get('question')
    answer = get_answer(client_id, model, question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
