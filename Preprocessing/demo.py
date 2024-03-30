import re
import nltk 
import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 

df = pd.read_csv('data.csv')

def preprocess(text): 
    if pd.isnull(text):  
        return ''
    
    tokens = nltk.word_tokenize(str(text).lower())  
    stop_words = set(stopwords.words('english')) 
    tokens = [token for token in tokens if token not in stop_words] 
    lemmatizer = WordNetLemmatizer() 
    tokens = [lemmatizer.lemmatize(token) for token in tokens] 
    text = ' '.join(tokens) 
    return text 


df['Symptoms'] = df['Symptoms'].apply(preprocess)


def extract_symptoms(text):
    symptoms_match = re.search(r'\b(symptoms?:?|manifestations?:?|signs?:?)\s*(.*?)\b', text, re.IGNORECASE)
    if symptoms_match:
        return symptoms_match.group(2).strip()
    else:
        return ''
df['Meaningful_Symptoms'] = df['Symptoms'].apply(extract_symptoms)
df['Has_Disease'] = 'Yes'
df[['Diseases', 'Meaningful_Symptoms', 'Has_Disease']].to_csv('processed_data.csv', index=False)



