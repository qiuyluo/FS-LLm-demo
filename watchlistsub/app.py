from flask import Flask, render_template, request, redirect, url_for, flash
import os
from model_utils import train_model, run_model, get_model_list

app = Flask(__name__)
app.secret_key = 'demo' 
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    app.logger.info("Home page accessed")
    return render_template('index.html')

@app.route('/client1')
def client1():
    return render_template('client.html', client_name='Client 1')

@app.route('/client2')
def client2():
    return render_template('client.html', client_name='Client 2')

@app.route('/train', methods=['POST'])
def train():
    train_model()
    flash("Training complete!")
    return redirect(url_for('home'))

@app.route('/upload_json', methods=['POST'])
def upload_json():
    file = request.files['json_file']
    client = request.form['client']
    if file:
        filename = f"{client.lower().replace(' ', '_')}_data.json"
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        flash(f"{client} dataset uploaded!")
    return redirect(url_for('client1' if client == 'Client 1' else 'client2'))

@app.route('/ask_questions', methods=['GET', 'POST'])
def ask_questions():
    if request.method == 'POST':
        model_name = request.form['model_name']
        question = request.form['question']
        answer = run_model(model_name, question)
        return render_template('ask_questions.html', models=get_model_list(), answer=answer)
    return render_template('ask_questions.html', models=get_model_list(), answer=None)

if __name__ == '__main__':
    app.run(debug=True)