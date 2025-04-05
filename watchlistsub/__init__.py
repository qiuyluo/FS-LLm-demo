from flask import Flask

app = Flask(__name__)
app.secret_key = 'demo'  

from watchlistsub import views