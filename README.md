# Stroke Prediction

## Description
- Data science and health is a field that increasingly merges as time goes on. This project is a glimpse at the capability machine learning models have in predicting stroke risk. The files in this repo contain the work, modules, and report that walks through the data science pipeline, resulting in a classification model used to predist stroke risk with a 74% recall.

## Goals
- The project aims to create a model that identifies individuals with a high risk of stroke based on stroke data. 

## Initial Questions
1. What does stroke look like in the dataset?
2. Is there a relationship between stroke and age?
3. Is there a relationship between stroke and gender?
4. Is there a relatio nship between blood sugar level and stroke?

## Plan
- Acquire data
- Prepare, clean, & split data
- Explore the data to find drivers and answer intital questions
- Create a model 
- Evaluate
- Conclude with recommendations and next steps

## Data Dictionary
| Feature | Definition | 
|:--------|:-----------|
| id | unique identifier |
| gender | "Male", "Female" or "Other" |
| age | age of the patient |
| hypertension | 0 if the patient doesn't have hypertension, 1 if the patient has hypertension |
| heart_disease | 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease |
| ever_married | "No" or "Yes" |
| work_type | "children", "Govt_jov", "Never_worked", "Private" or "Self-employed" |
| Residence_type | "Rural" or "Urban" |
| avg_glucose_level | average glucose level in blood |
| bmi | body mass index |
| smoking_status | "formerly smoked", "never smoked", "smokes" or "Unknown" |
| stroke | 1 if the patient had a stroke or 0 if not | 


## Steps to Reproduce
1. Clone this repo
2. Go to https://kaggle.com/me/account (sign in if required).
3. Scroll down to the "API" section and click "Create New API Token". This will download a file kaggle.json 
4. Move kaggle.json file to the same directory/folder as the 'final_report.ipynb' notebook
5. Run the 'final_report.ipynb' notebook

## Takeaways
* Stroke represented roughly 5% of the data which influenced the decision to oversample to accomodate an imbalanced dataset
* Demographically, only age had a significant relationship to stroke, while gender's independence could not be rejected
* Average blood sugar level was found to have a statistically significant relationship to stroke
* On test, the VotingClassifier model performed with a 74% recall and a 66% accuracy.

## Recommendations
* Acquire more health and demographic data
* Increase the robustness of the smoking_status data
* Use this model as a preliminary screening tool to asses stroke risk
