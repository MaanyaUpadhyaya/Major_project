import pandas as pd
import spacy

df = pd.read_csv('data.csv')

nlp = spacy.load('en_core_web_sm')

def extract_symptoms(text):
    if pd.isnull(text):
        return set()
    doc = nlp(text.lower()) 
    symptoms = set()
    for ent in doc.ents:
        if ent.label_ == 'SYMPTOM':
            symptoms.add(ent.text)
    return symptoms

df['Symptoms'] = df['Symptoms'].apply(lambda x: extract_symptoms(x) if pd.notnull(x) else set())

all_symptoms = set(symptom for symptoms in df['Symptoms'] for symptom in symptoms)
for symptom in all_symptoms:
    df[symptom] = df['Symptoms'].apply(lambda x: 1 if symptom in x else 0)

df['Disease_Present'] = df['Diseases'].apply(lambda x: 1 if pd.notnull(x) else 0)
df.to_csv('ready_data.csv', index=False)




