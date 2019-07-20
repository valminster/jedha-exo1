from flask import Flask, request
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump, load
import json
from utils import parse_box_office

app = Flask(__name__)


@app.route('/get_movie_metadata', methods=['POST'])
def get_movie_metadata():
    movies = json.loads(request.form['movies'])
    return json.dumps([
        requests.get('http://www.omdbapi.com/?t={}&apikey=952090a1'.format(movie)).json()
        for movie in movies
    ])


@app.route('/train_model', methods=['POST'])
def train_model():
    model_name = request.form['model_name']
    data = json.loads(request.form['data'])
    df = pd.DataFrame(data)

    df['BoxOffice'] = df['BoxOffice'].apply(parse_box_office).dropna()
    df['USA?'] = df['Country'].apply(lambda country: 1 if (country == 'USA') else 0)
    df['Genre'] = df['Genre'].apply(lambda genre: genre.split(',')[0])
    df = pd.get_dummies(df, columns=['Genre'], drop_first=True)
    df = df.dropna()

    X = df[['BoxOffice', 'USA?'] + [col for col in df.columns if 'Genre_' in col]]
    y = df['imdbRating'].astype(float)

    reg = LinearRegression()

    reg.fit(X, y)

    dump(reg, model_name)

    return 'SUCCESS'


@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model_name']
    data = json.loads(request.form['data'])
    df = pd.DataFrame(data)

    return json.dumps(load(model_name).predict(df).tolist())



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

