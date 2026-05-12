from flask import Flask, request, jsonify, render_template
import os
import pickle
import numpy as np

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'scaler.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

REQUIRED_FEATURES = [
    'StudyTimeWeekly',
    'Absences',
    'Tutoring'
]

def performance_category(prediction):
    if prediction >= 3.5:
        return 'Excellent'
    if prediction >= 3.0:
        return 'Good'
    if prediction >= 2.5:
        return 'Average'
    return 'Needs Improvement'


def build_suggestions(study_time, absences_val, tutoring_val):
    suggestions = []

    if study_time < 8:
        suggestions.append('Increase study time by 2–3 hours per week to help lift GPA.')
    if absences_val > 5:
        suggestions.append('Reduce absences to 5 or fewer sessions to support more stable progress.')
    if int(tutoring_val) == 0:
        suggestions.append('Consider using tutoring to strengthen understanding and boost results.')

    if not suggestions:
        suggestions.append('You’re already performing well. Maintain consistency and keep your routine steady.')

    return suggestions


def feature_impact():
    return [
        {
            'title': 'Study Time Weekly',
            'impact': 'Positive',
            'icon': '📈',
            'detail': 'Increasing weekly study time tends to improve predicted GPA.'
        },
        {
            'title': 'Absences',
            'impact': 'Negative',
            'icon': '📉',
            'detail': 'Higher absence counts usually correlate with lower predicted performance.'
        },
        {
            'title': 'Tutoring',
            'impact': 'Positive',
            'icon': '📈',
            'detail': 'Tutoring support is associated with more consistent score improvements.'
        }
    ]


def build_whatif_message(category, current_gpa, gpa_diff, percentage_diff):
    if category == 'Excellent':
        headline = 'Excellent work already — keep refining your routine.'
        description = 'A few small changes can help you sustain top performance while reducing stress.'
    elif category == 'Good':
        headline = 'Good performance with a clear path to excellence.'
        description = 'Increasing study time and improving attendance can lift you into the excellent range.'
    elif category == 'Average':
        headline = 'Average performance can improve with consistent effort.'
        description = 'Stronger study habits, fewer absences, and tutoring support will boost your progress.'
    else:
        headline = 'Needs more support, but improvement is possible.'
        description = 'Focus on weekly study, attendance, and tutoring to achieve a meaningful GPA increase.'

    action = f'Projected gain: +{gpa_diff:.2f} GPA and +{percentage_diff:.2f}%.'
    return {
        'headline': headline,
        'description': description,
        'action': action
    }


@app.route('/')
def home():
    return render_template(
        'index.html',
        prediction_text='',
        percentage_text='',
        category='',
        study='',
        absences='',
        tutoring='',
        gpa_value=0,
        percentage_value=0,
        suggestions=[],
        feature_impact=feature_impact(),
        improved_gpa=None,
        improved_percentage=0,
        gpa_diff=0,
        percentage_diff=0,
        whatif_info=None
    )

@app.route('/predict_form', methods=['POST'])
def predict_form():
    study = request.form.get('StudyTimeWeekly', '')
    absences = request.form.get('Absences', '')
    tutoring = request.form.get('Tutoring', '')

    try:
        study_time = float(study)
        absences_val = float(absences)
        tutoring_val = float(tutoring)
    except (ValueError, TypeError):
        return render_template(
            'index.html',
            prediction_text='Please enter valid numeric values for all fields.',
            percentage_text='',
            category='',
            study=study,
            absences=absences,
            tutoring=tutoring,
            gpa_value=0,
            percentage_value=0,
            suggestions=[],
            feature_impact=feature_impact(),
            improved_gpa=None,
            improved_percentage=0,
            gpa_diff=0,
            percentage_diff=0
        )

    raw_features = np.array([[study_time, absences_val, tutoring_val]])
    scaled_features = scaler.transform(raw_features)
    prediction = model.predict(scaled_features)[0]
    current_gpa = prediction
    current_percentage = (current_gpa / 4) * 100
    category = performance_category(current_gpa)

    improved_study = study_time + 2
    improved_absences = max(0, absences_val - 2)
    improved_tutoring = 1
    improved_features = np.array([[improved_study, improved_absences, improved_tutoring]])
    improved_scaled = scaler.transform(improved_features)
    improved_gpa = model.predict(improved_scaled)[0]
    improved_percentage = (improved_gpa / 4) * 100
    gpa_diff = improved_gpa - current_gpa
    percentage_diff = improved_percentage - current_percentage

    suggestions = build_suggestions(study_time, absences_val, tutoring_val)
    impacts = feature_impact()
    whatif_info = build_whatif_message(category, current_gpa, gpa_diff, percentage_diff)

    return render_template(
        'index.html',
        prediction_text=f'Predicted GPA: {current_gpa:.2f}',
        percentage_text=f'Predicted Percentage: {current_percentage:.2f}%',
        category=category,
        study=study,
        absences=absences,
        tutoring=tutoring,
        gpa_value=round(current_gpa, 2),
        percentage_value=round(current_percentage, 2),
        suggestions=suggestions,
        feature_impact=impacts,
        improved_gpa=round(improved_gpa, 2),
        improved_percentage=round(improved_percentage, 2),
        gpa_diff=round(gpa_diff, 2),
        percentage_diff=round(percentage_diff, 2),
        whatif_info=whatif_info
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    missing = [key for key in REQUIRED_FEATURES if key not in data]
    if missing:
        return jsonify({'error': 'Missing fields', 'missing': missing}), 400

    try:
        features = np.array([float(data[key]) for key in REQUIRED_FEATURES]).reshape(1, -1)
        scaled_features = scaler.transform(features)
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid input values. All fields must be numeric.'}), 400

    prediction = model.predict(scaled_features)[0]
    category = performance_category(prediction)
    return jsonify({'predicted_gpa': float(prediction), 'category': category})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
