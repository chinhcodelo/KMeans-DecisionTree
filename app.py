from flask import Flask, render_template, request
import pandas as pd
from model.kmeans import run_kmeans
from model.decision_tree import run_decision_tree
from model.kmeans_decision_tree import run_kmeans_decision_tree

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    uploaded_file = request.files['file']
    algorithm = request.form['algorithm']

    if uploaded_file.filename != '':
        df = pd.read_csv(uploaded_file)

        if algorithm == 'kmeans':
            img_path = run_kmeans(df)
        elif algorithm == 'decision_tree':
            img_path = run_decision_tree(df)
        elif algorithm == 'kmeans_decision_tree':
            img_path = run_kmeans_decision_tree(df)
        else:
            return "Thuật toán không hợp lệ."

        return render_template('result.html', img_path=img_path, algo=algorithm)
    else:
        return "Vui lòng tải lên tập tin CSV."

if __name__ == '__main__':
    app.run(debug=True)
