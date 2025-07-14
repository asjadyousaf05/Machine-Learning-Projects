import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Apply custom styles
st.markdown("""
    <style>
    body, .stApp {
        background-color: #f0f2f6;
        color: #000000 !important;
    }

    h1, h2, h3, h4, h5, h6, label, .stTextInput label, .stMarkdown p {
        color: #000000 !important;
    }

    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #000000;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #6C63FF;
    }

    .stButton>button {
        background-color: #6C63FF;
        color: white;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #4e49c0;
    }
    </style>
""", unsafe_allow_html=True)


# Load models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        '🩺 Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# ------------------ Diabetes Page ------------------
if selected == 'Diabetes Prediction':
    st.title('🔬 Diabetes Prediction using ML')
    st.subheader("Enter the patient information:")

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.text_input('👶 Pregnancies', placeholder="e.g. 2")
        BloodPressure = st.text_input('🩸 Blood Pressure', placeholder="e.g. 72")
        Insulin = st.text_input('💉 Insulin Level', placeholder="e.g. 94")
        DiabetesPedigreeFunction = st.text_input('🧬 Pedigree Function', placeholder="e.g. 0.47")
    with col2:
        Glucose = st.text_input('🍬 Glucose Level', placeholder="e.g. 120")
        SkinThickness = st.text_input('📏 Skin Thickness', placeholder="e.g. 23")
        BMI = st.text_input('⚖️ BMI', placeholder="e.g. 28.1")
        Age = st.text_input('🎂 Age', placeholder="e.g. 45")

    diab_diagnosis = ''
    if st.button('🔍 Diabetes Test Result'):
        try:
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                          float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            result = diabetes_model.predict([input_data])
            diab_diagnosis = '🟥 The person is diabetic.' if result[0] == 1 else '🟩 The person is not diabetic.'
        except:
            diab_diagnosis = '⚠️ Please enter valid numeric values only.'

    st.success(diab_diagnosis)

# ------------------ Heart Disease Page ------------------
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')
    st.subheader("Provide the heart health indicators:")

    col1, col2 = st.columns(2)
    with col1:
        age = st.text_input('🎂 Age')
        sex = st.text_input('⚧️ Sex (1=Male, 0=Female)')
        cp = st.text_input('💢 Chest Pain Type (0–3)')
        trestbps = st.text_input('🩸 Resting BP')
        chol = st.text_input('🧪 Serum Cholesterol')
        fbs = st.text_input('🧃 Fasting Blood Sugar > 120 (1/0)')
        restecg = st.text_input('📉 Resting ECG (0–2)')
    with col2:
        thalach = st.text_input('🏃 Max Heart Rate')
        exang = st.text_input('😮 Exercise-induced Angina (1/0)')
        oldpeak = st.text_input('📉 ST Depression')
        slope = st.text_input('📈 Slope of ST Segment')
        ca = st.text_input('🫀 Major Vessels Colored')
        thal = st.text_input('🧬 Thal (1=fixed, 2=reversible, 3=normal)')

    heart_diagnosis = ''
    if st.button('🔍 Heart Disease Test Result'):
        try:
            input_data = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs),
                          float(restecg), float(thalach), float(exang), float(oldpeak),
                          float(slope), float(ca), float(thal)]
            result = heart_disease_model.predict([input_data])
            heart_diagnosis = '🟥 The person has heart disease.' if result[0] == 1 else '🟩 The person does not have heart disease.'
        except:
            heart_diagnosis = '⚠️ Please enter valid numeric values only.'

    st.success(heart_diagnosis)

# ------------------ Parkinson's Page ------------------
if selected == 'Parkinsons Prediction':
    st.title("🧠 Parkinson's Disease Prediction using ML")
    st.subheader("Voice and frequency indicators:")

    inputs = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    values = []
    cols = st.columns(3)
    for i, label in enumerate(inputs):
        with cols[i % 3]:
            values.append(st.text_input(label, placeholder="e.g. 119.99"))

    parkinsons_diagnosis = ''
    if st.button("🔍 Parkinson's Test Result"):
        try:
            input_data = [float(i) for i in values]
            result = parkinsons_model.predict([input_data])
            parkinsons_diagnosis = "🟥 The person has Parkinson's disease." if result[0] == 1 else "🟩 The person does not have Parkinson's disease."
        except:
            parkinsons_diagnosis = "⚠️ Please enter valid numeric values only."

    st.success(parkinsons_diagnosis)
