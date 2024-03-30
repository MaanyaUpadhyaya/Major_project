import pandas as pd
import spacy
from spacy.tokens import Doc
nlp = spacy.load('en_core_web_sm')
df = pd.read_csv('data.csv')
#print((df['Symptoms']))
def preprocess(text):
    if pd.isnull(text):  # Check for NaN values
        return ''
    doc = nlp(str(text))
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    text = ' '.join(tokens) 
    return text


df['Symptoms'] = df['Symptoms'].apply(preprocess)
import re
def extract_symptoms(text):
    symptom_patterns = [
        r'\b(symptoms?:?|manifestations?:?|signs?:?)\s*(.*?)\b',
        r'\b(issues?:?|complaints?:?|problems?:?)\s*(.*?)\b',
        r'\b(affected by|experiencing|feeling)\s*(.*?)\b',
        r'\b(notice|observe|feel)\s*(.*?)\b',
        r'\b(fever|temperature)\s*(.*?)\b',
        r'\b(cough)\s*(.*?)\b',
        r'\b(headache)\s*(.*?)\b',
        r'\b(fatigue|tiredness)\s*(.*?)\b',
        r'\b(pain)\s*(.*?)\b',
        r'\b(nausea)\s*(.*?)\b',
        r'\b(dizziness)\s*(.*?)\b',
        r'\b(shortness of breath)\s*(.*?)\b',
        r'\b(sore throat)\s*(.*?)\b',
        r'\b(chest pain)\s*(.*?)\b',
        r'\b(abdominal pain)\s*(.*?)\b',
        r'\b(rash)\s*(.*?)\b',
        r'\b(joint pain)\s*(.*?)\b',
        r'\b(muscle weakness)\s*(.*?)\b',
        r'\b(tingling)\s*(.*?)\b',
        r'\b(gastrointestinal tract)\s*(.*?)\b',
        r'\b(inflammatory bowel disease|IBD)\s*(.*?)\b',
        r'\b(cancer)\s*(.*?)\b',
        r'\b(growth)\s*(.*?)\b',
        r'\b(adolescence)\s*(.*?)\b',
        r'\b(adulthood)\s*(.*?)\b',
        r'\b(infections)\s*(.*?)\b',
        r'\b(exposure)\s*(.*?)\b',
        r'\b(gastritis)\s*(.*?)\b',
        r'\b(gastroparesis)\s*(.*?)\b',
        r'\b(different symptoms)\s*(.*?)\b',
        r'\b(respiratory symptom cluster)\s*(.*?)\b',
        r'\b(infectious disease)\s*(.*?)\b',
        r'\b(herpes viruses)\s*(.*?)\b',
        r'\b(Mono)\s*(.*?)\b',
        r'\b(blood tests)\s*(.*?)\b',
        r'\b(early signs and symptoms)\s*(.*?)\b',
        r'\b(15 years or more before dementia develops)\s*(.*?)\b',
        r'\b(constipation)\s*(.*?)\b',
        r'\b(dizziness)\s*(.*?)\b',
    ]
    all_symptoms = []
    for symptom_pattern in symptom_patterns:
        match = re.search(symptom_pattern, text, re.IGNORECASE)
        if match:
            all_symptoms.extend(match.group(2).strip().split(','))
    return ', '.join(all_symptoms)

df['Meaningful_Symptoms'] = df['Symptoms'].apply(extract_symptoms)
df['Has_Disease'] = 'Yes'
df[['Diseases', 'Meaningful_Symptoms', 'Has_Disease']].to_csv('processed_data.csv', index=False)
