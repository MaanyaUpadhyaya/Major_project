from flask import Flask, render_template, request, jsonify
import numpy as np
from model_training import ektra7at, discrp, rnd_forest,df1

app = Flask(__name__)

def predd(x, *symptoms):
    pysymptoms = symptoms
    psymptoms = pysymptoms[0]
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j]==a[k]:
                psymptoms[j]=b[k]
    psy = [psymptoms]
    psy = psy[0][:17]
    num_rows = 1  
    num_cols = 17 
    psy1 = [psy[i:i+num_cols] for i in range(0, len(psy), num_cols)]
    pred2 = x.predict(psy1)
    disp= discrp[discrp['Disease']==pred2[0]]
    disp = disp.values[0][1]
    recomnd = ektra7at[ektra7at['Disease']==pred2[0]]
    c=np.where(ektra7at['Disease']==pred2[0])[0][0]
    precuation_list=[]
    for i in range(1,len(ektra7at.iloc[c])):
          precuation_list.append(ektra7at.iloc[c,i])
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(" ")
    print("The Disease Name: ",pred2[0])
    print(" ")
    print(" ")
    print("The Disease Discription: ",disp)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(" ")
    print("!!Recommended Things to do at home: !!")
    for i in precuation_list:
        print(i)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        selected_symptoms = request.form.getlist("selectedSymptoms")
        new_list = []
        if isinstance(selected_symptoms[0], str):
            new_list = selected_symptoms[0].split(',')
        modified_list = [new_list] + [0] * (17 - len(selected_symptoms))
        symp = []
        symp.extend(modified_list[0])
        symp.extend(modified_list[1:])
        predd(rnd_forest, symp)
        return jsonify({"message": "Prediction completed.Please check the console for the diseases and precautions. KINDLY REMEMBER YOU NEED TO CONSULT A DOCTOR FOR THE BEST REVIEW OF THE CONCERENED DISEASES"})

    return render_template("index.html", symptoms=df1["Symptom"])

if __name__ == "__main__":
    app.run(debug=True)
