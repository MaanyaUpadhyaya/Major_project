from flask import Flask, render_template, request
import pandas as pd
import spacy
from disease_prediction import DiseasePredictor

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('Datasets/Testing.csv')

# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Initialize DiseasePredictor
predictor = DiseasePredictor()

@app.route('/')
def index():
    return render_template('index.html', bot_response=["Welcome! How can I assist you today?"])

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']
    symptoms = extract_symptoms(user_message)
    if symptoms:
        predicted_disease = predictor.predict_disease(symptoms)
        bot_response = f"The predicted disease based on symptoms is: {predicted_disease}"
    else:
        bot_response = "No symptoms detected. Please provide symptoms for prediction."
    return render_template('index.html', user_message=user_message, bot_response=bot_response)

def extract_symptoms(user_message):
    # Process the user's message using spaCy
    doc = nlp(user_message)
    # Extract nouns (likely symptoms) from the processed text
    symptoms = [token.text.lower() for token in doc if token.pos_ == 'NOUN']
    return symptoms

if __name__ == '__main__':
    app.run(debug=True)
