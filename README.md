# PreScienceMED : A Medical Chatbot

## Abstract
The project is motivated by the urgent need within the healthcare domain to effectively harness the vast amount of unstructured digital health data. In response to this challenge, the project focuses on the application of automatic text classification techniques, leveraging advancements in natural language processing (NLP) to facilitate precise disease prediction.Central to the project is the utilization of machine learning algorithms and domain-specific knowledge bases to categorize and organize medical text data. Through a systematic approach encompassing data collection, preprocessing, feature extraction, model training, and evaluation, the project aims to develop robust text classification models tailored to the nuances of medical language and context.The significance of this research lies in its potential to enhance healthcare decision-making and improve patient outcomes. 

## Requirements

| Software requirements| Description                           |
|----------------------|---------------------------------------|
| Language             | Python 3                              |
| Environment          | Visual Studio Code                    |
| Packages             | NumPy,MatPlotlib,Flask,Joblib         |



| Hardware requirements| Description                           |
|----------------------|---------------------------------------|
| Processor            |  Intel core i5 2.40Gh and above       |
| RAM                  |  8GB or above                         |
| Operating System     |  Windows 7 or above, Ubuntu,MacOS     |
| Disk Space           | Minimum 5GB                           |


## Installation
1. Clone this repository to your local machine or as for a pull request
2. Once the code is on your local machine, please maake sure all the datasets are available in the directory selected
3. Please make sure you have all the modules and dependencies and then run the code
4. TO RUN :- python flask_frontend.py or python3 flask_frontend.py

## Methodology 
1. Data collection
2. Data preprocessing
3. Model training
4. Model evaluation
5. Model deployment

## Models Considered
1. Descion Tree Classifier
2. Random Forest Classifier
3. SVM
4. Guassian NB Classifier


| Model name           | Accuracy | F1 Score |
|----------------------|----------|----------|
| Desicion Tree        |          |          |
| Randon forest        |          |          |
| SVM                  |          |          |
| Guassian NB          |          |          |

Out of these model the Random Forest Classifier gave the best results with the accuracy of 99.59% and thus the same was used in model training

## Model deployment

The model was deployed into a web application using Flask. In FLask we can render an HTML file for the web app and produce an attractive forntend and append this to the trained model in the backend. With the use of Javascript, we could get the symptoms entered by the user to the model for the presice disease prediction.



