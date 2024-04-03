from flask import Flask, render_template, request, jsonify
import logging
import joblib
from model_tryouts import ektra7at, loaded_rf, discrp, rnd_forest

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def process_symptoms(selected_symptoms, all_symptoms, model):
    # Initialize the feature vector with zeros
    feature_vector = [0] * len(all_symptoms)

    # Iterate through selected symptoms and set corresponding entries to 1
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            feature_vector[index] = 1  # Set the corresponding entry to 1

    # Use the trained model to get feature importance scores
    feature_importances = model.feature_importances_

    # Sort symptoms based on feature importance scores
    sorted_symptoms = [symptom for _, symptom in sorted(zip(feature_importances, all_symptoms), reverse=True)]

    # Select top 17 symptoms based on importance scores
    selected_features = sorted_symptoms[:17]

    # Create a subset feature vector containing only the selected features
    subset_feature_vector = [feature_vector[all_symptoms.index(symptom)] for symptom in selected_features]

    return subset_feature_vector


@app.route("/", methods=["GET", "POST"])
def home():
    symptoms = [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
        'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
        'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets',
        'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level',
        'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion',
        'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
        'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
        'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
        'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
        'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
        'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
        'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
        'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
        'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
        'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance',
        'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_ofurine',
        'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
        'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
        'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
        'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
        'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
        'distention_of_abdomen', 'history_of_alcohol_consumption', 'blood_in_sputum', 'prominent_veins_on_calf',
        'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
        'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
        'yellow_crust_ooze'
    ]
    if request.method == "POST":
        selected_symptoms = request.form.getlist("selectedSymptoms")
        logger.debug("Selected symptoms: %s", selected_symptoms)

        # Process the symptoms
        processed_symptoms = process_symptoms(selected_symptoms, symptoms, rnd_forest)
        logger.debug("Processed symptoms: %s", processed_symptoms)

        # Make predictions using the loaded model
        predictions = rnd_forest.predict([processed_symptoms])
        logger.debug("Predictions: %s", predictions)

        # Get disease description and precautions
        disease_description = discrp[discrp['Disease'] == predictions[0]]['Description'].values[0]
        precautions = ektra7at[ektra7at['Disease'] == predictions[0]].iloc[0, 1:].values.tolist()

        # Return predictions and additional information as JSON response
        return jsonify({
            "predicted_disease": predictions[0],
            "disease_description": disease_description,
            "precautions": precautions
        })

    return render_template("index.html", symptoms=symptoms)


if __name__ == "__main__":
    app.run(debug=True)
