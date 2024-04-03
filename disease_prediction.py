import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

class DiseasePredictor:
    def __init__(self):
        self.models = {
            'DecisionTree': DecisionTreeClassifier(),
            'GaussianNB' : GaussianNB(),
            'Random_forest': RandomForestClassifier(),
        }
        self.label_encoder = LabelEncoder()
        training_data_path = 'Datasets/Training.csv'
        self.training = pd.read_csv(training_data_path)
        self.cols = self.training.columns[:-1]
        self.train_models()

    def train_models(self):
        X = self.training[self.cols]
        y = self.training['prognosis']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            # Save the model as a pickle file
            joblib.dump(model, f'{name}_model.pkl')
        self.label_encoder.fit(y)

    def predict_disease(self, symptoms):
        encoded_symptoms = self.label_encoder.transform(symptoms)
        predictions = {}
        for name, model in self.models.items():
            # Load the model from the file
            model = joblib.load(f'{name}_model.pkl')
            encoded_symptoms_2d = encoded_symptoms.reshape(1, -1)
            prediction = model.predict(encoded_symptoms_2d)
            predictions[name] = self.label_encoder.inverse_transform(prediction)[0]
        return predictions
