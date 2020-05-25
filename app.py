from flask import Flask, request
from flask_restful import Resource, Api
import os
import torch
import spacy
import pickle
from sentimentAnalysis import CNN, SentimentAnalysis

# Init app
app = Flask(__name__)
api = Api(app)

# Init device and fetch params for model initialization
nlp = spacy.load('en')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sentimentAnalysis = SentimentAnalysis()

@app.route('/predict', methods=['POST'])
def predict():
    req_data = request.get_json()
    model_params, labels, vocab = sentimentAnalysis.get_model_params()
    model = CNN(**model_params)
    if req_data['sentence']:
        model.load_state_dict(torch.load('model/sentimentModel.pt'))
        model.eval()
        sentence = req_data['sentence']
        pred_class = sentimentAnalysis.predict_class(model, sentence, vocab)
        return labels[pred_class]
    else:
        return 'Please provide a sentence to predict sentiment of (.json format with keyword: "sentence")'


@app.route('/train', methods=['GET'])
def train():
    return 'not yet implemented'



# Run Server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')