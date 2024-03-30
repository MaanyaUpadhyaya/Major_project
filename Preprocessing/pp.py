import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Your existing code to read the CSV file and create binary columns
df = pd.read_csv('data.csv')
df['Symptoms'].fillna('', inplace=True)

symptoms_list = [
    "respiratory disease", "non-motor systems", "tremor", "rigidity", 
    "streptococcus pneumoniae", "lung diseases", "pathological conditions", 
    "gas exchange difficulty", "ALS", "external injury", "medical conditions", 
    "specific signs", "disease", "waterborne diseases", "virulence factors", 
    "symptoms and complications", "COVID-19", "motor neuron diseases", 
    "stiff muscles", "muscle twitches", "gradual increasing weakness", 
    "muscle wasting", "diarrhea", "vomiting", "waterborne illness", "skin", 
    "ear", "respiratory", "lyme disease", "respiratory symptom cluster", 
    "Parkinson's disease", "motor neuron diseases", "sexually transmitted infection", 
    "sexually transmitted diseases", "infections", "fever", "headaches", 
    "tiredness", "loss of ability to move", "hand, foot, and mouth disease", 
    "varied", "Alzheimer's disease", "variant creutzfeldt–jakob disease", 
    "coxsackievirus", "enterovirus", "neurological disorder", "neurological symptoms", 
    "dementia", "difficulty in remembering recent events", "problems with language"
]

for symptom in symptoms_list:
    df[symptom] = df['Symptoms'].str.contains(symptom, case=False)
df.to_csv('processed_data.csv', index=False)

for symptom in symptoms_list:
    df[symptom] = df['Symptoms'].apply(lambda x: symptom.lower() in x.lower())
numerical_df = df[symptoms_list].astype(int)
numerical_df.to_csv('numerical_data.csv', index=False)
