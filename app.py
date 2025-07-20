from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from geopy.distance import geodesic

app = Flask(__name__)

# Load datasets
hospitals_df = pd.read_csv('datasets/hospitals.csv')
diseases_df = pd.read_csv('datasets/diseases.csv')

# Convert age columns to integers
diseases_df['min_age'] = diseases_df['min_age'].astype(int)
diseases_df['max_age'] = diseases_df['max_age'].astype(int)

# Dynamically create disease_specialty_map from diseases.csv
disease_specialty_map = dict(zip(diseases_df['disease'], 
                                 zip(diseases_df['specialization'], diseases_df['severity'])))

# Dynamically create valid_symptoms from diseases.csv
valid_symptoms = set()
for symptoms in diseases_df['symptoms']:
    valid_symptoms.update([s.strip().lower() for s in symptoms.split(',')])

# Prepare disease-symptom data for KNN
mlb = MultiLabelBinarizer(classes=sorted(valid_symptoms))
symptom_matrix = mlb.fit_transform(diseases_df['symptoms'].apply(lambda x: [s.strip().lower() for s in x.split(',')]))
disease_labels = diseases_df['disease'].values

def preprocess_symptoms(user_symptoms):
    if not user_symptoms or not user_symptoms.strip():
        return [], {}
    symptom_list = [s.strip().lower() for s in user_symptoms.split(',')]
    cleaned_symptoms = []
    invalid_symptoms = []
    suggestions = {}

    for symptom in symptom_list:
        if symptom in valid_symptoms:
            cleaned_symptoms.append(symptom)
        else:
            invalid_symptoms.append(symptom)
            matches = [s for s in valid_symptoms if s.startswith(symptom[:3])]
            suggestions[symptom] = [next((s_orig for s_orig in diseases_df['symptoms'].str.split(',').explode() 
                                          if s_orig.strip().lower() == m), m) for m in matches[:3]] or ["No close matches"]

    return cleaned_symptoms, suggestions

def predict_diseases(user_symptoms, user_age, user_gender, top_n=3):
    cleaned_symptoms, suggestions = preprocess_symptoms(user_symptoms)
    if not cleaned_symptoms and suggestions:
        return None, None, f"Invalid symptoms detected. Suggestions: {suggestions}"
    if not cleaned_symptoms:
        return None, None, "No valid symptoms provided."

    user_symptom_vector = mlb.transform([cleaned_symptoms])[0]
    knn = NearestNeighbors(n_neighbors=len(disease_labels), metric='hamming').fit(symptom_matrix)
    distances, indices = knn.kneighbors([user_symptom_vector])

    # Adjust distances based on age and gender
    disease_distances = []
    for dist, idx in zip(distances[0], indices[0]):
        disease = disease_labels[idx]
        min_age = diseases_df.loc[idx, 'min_age']
        max_age = diseases_df.loc[idx, 'max_age']
        gender_prev = diseases_df.loc[idx, 'gender_prevalence']
        adjusted_dist = dist

        if user_age < min_age or user_age > max_age:
            adjusted_dist += 0.2
        if gender_prev != 'both' and gender_prev != user_gender:
            adjusted_dist += 0.2

        disease_distances.append((disease, adjusted_dist))

    # Sort by adjusted distance
    disease_distances.sort(key=lambda x: x[1])
    top_diseases = disease_distances[:top_n]

    # Calculate confidence based on adjusted distances
    total_weight = sum(1 / (adj_dist + 1e-5) for disease, adj_dist in top_diseases)
    predictions = [
        (disease, f"Confidence: {(1 / (adj_dist + 1e-5) / total_weight):.2%}")
        for disease, adj_dist in top_diseases
    ]
    return predictions, cleaned_symptoms[0], None

def find_nearest_hospitals(user_lat, user_lon, disease, severity, k=3):
    specialty = disease_specialty_map.get(disease, ('General', 'Mild'))[0]
    eligible_hospitals = hospitals_df[hospitals_df['specialization'].str.contains(specialty, case=False, na=False)].copy()

    if eligible_hospitals.empty:
        eligible_hospitals = hospitals_df.copy()

    eligible_hospitals['distance'] = eligible_hospitals.apply(
        lambda row: geodesic((user_lat, user_lon), (row['latitude'], row['longitude'])).miles, axis=1
    )

    eligible_hospitals = eligible_hospitals.sort_values(by=['distance', 'rating'], ascending=[True, False])

    recommendations = eligible_hospitals.head(k).copy()
    recommendations['maps_link'] = recommendations.apply(
        lambda row: f"https://www.google.com/maps/dir/?api=1&destination={row['latitude']},{row['longitude']}", axis=1
    )

    return recommendations.to_dict(orient='records')

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    recommendations = []
    error_message = None
    severity = None

    if request.method == 'POST':
        user_symptoms = request.form.get('symptoms', '').strip()
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        age = request.form.get('age')
        gender = request.form.get('gender')

        if not latitude or not longitude:
            error_message = "Location access denied. Please enable GPS."
        elif not age or not gender:
            error_message = "Please provide age and gender."
        else:
            user_lat, user_lon = float(latitude), float(longitude)
            user_age = int(age)
            user_gender = gender.lower()

            if user_symptoms:
                predictions, top_disease, error_message = predict_diseases(user_symptoms, user_age, user_gender)
                if predictions:
                    top_disease_name = predictions[0][0]
                    severity = disease_specialty_map.get(top_disease_name, ('General', 'Mild'))[1]
                    recommendations = find_nearest_hospitals(user_lat, user_lon, top_disease_name, severity)

    return render_template('index.html', predictions=predictions, recommendations=recommendations,
                           error_message=error_message, severity=severity)

if __name__ == '__main__':
    app.run(debug=True)