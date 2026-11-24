
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Plateforme Deep Learning",
    page_icon="🧠",
    layout="wide"
)

MODEL_DIR = Path(__file__).parent / "models"


# =========================
# CHARGEMENT DES MODELES
# =========================

@st.cache_resource
def load_model(filename):
    return tf.keras.models.load_model(
        MODEL_DIR / filename,
        compile=False
    )
@st.cache_resource
def load_tb_model():
    model_path = hf_hub_download(
        repo_id="DouceR/tuberculosis-model",
        filename="vgg16_model_finetuned.h5"
    )

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )

@st.cache_resource
def load_bank_preprocessing():
    encoders = joblib.load(MODEL_DIR / "bank_encoders.joblib")
    scaler = joblib.load(MODEL_DIR / "bank_scaler.joblib")
    return encoders, scaler

@st.cache_resource
def load_student_preprocessing():
    encoder = joblib.load(MODEL_DIR / "student_encoder.joblib")
    scaler = joblib.load(MODEL_DIR / "student_scaler.joblib")
    return encoder, scaler



# =========================
# PRETRAITEMENT DES IMAGES
# =========================

def predict_image(model, image, size):
    image = image.convert("RGB")
    image = image.resize(size)

    array = np.array(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    return model.predict(array, verbose=0)[0]


# =========================
# INTERFACE PRINCIPALE
# =========================

st.title("🧠 Plateforme de Deep Learning")
st.write("Démonstration des modèles développés.")

projet = st.sidebar.selectbox(
    "Choisir un projet",
    [
        "Accueil",
        "Bank Marketing",
        "Student Performance",
        "CIFAR-10",
        "Pneumonia",
        "Tuberculosis"
    ]
)


# =========================
# ACCUEIL
# =========================

if projet == "Accueil":

    st.header("Bienvenue")

    st.write(
        "Cette plateforme regroupe nos différents projets "
        "de Deep Learning."
    )

    st.info("Choisissez un projet dans le menu à gauche.")


# =========================
# BANK MARKETING
# =========================

elif projet == "Bank Marketing":

    st.header("Prédiction de souscription bancaire")

    st.write(
        "Ce modèle prédit si un client est susceptible "
        "de souscrire à un dépôt bancaire."
    )

    try:
        encoders, scaler = load_bank_preprocessing()

        with st.form("bank_form"):

            st.subheader("Informations du client")

            col1, col2 = st.columns(2)

            with col1:

                age = st.number_input(
                    "Âge",
                    min_value=18,
                    max_value=100,
                    value=35
                )

                job = st.selectbox(
                    "Profession",
                    encoders["job"].classes_.tolist()
                )

                marital = st.selectbox(
                    "Situation matrimoniale",
                    encoders["marital"].classes_.tolist()
                )

                education = st.selectbox(
                    "Niveau d'éducation",
                    encoders["education"].classes_.tolist()
                )

                default = st.selectbox(
                    "Crédit en défaut ?",
                    encoders["default"].classes_.tolist()
                )

                housing = st.selectbox(
                    "Crédit immobilier ?",
                    encoders["housing"].classes_.tolist()
                )

                loan = st.selectbox(
                    "Crédit personnel ?",
                    encoders["loan"].classes_.tolist()
                )

                contact = st.selectbox(
                    "Type de contact",
                    encoders["contact"].classes_.tolist()
                )

            with col2:

                month = st.selectbox(
                    "Mois du dernier contact",
                    encoders["month"].classes_.tolist()
                )

                day_of_week = st.selectbox(
                    "Jour de la semaine",
                    encoders["day_of_week"].classes_.tolist()
                )

                duration = st.number_input(
                    "Durée du dernier contact (secondes)",
                    min_value=0,
                    max_value=10000,
                    value=180
                )

                campaign = st.number_input(
                    "Nombre de contacts pendant cette campagne",
                    min_value=1,
                    max_value=100,
                    value=1
                )

                pdays = st.number_input(
                    "Jours depuis le dernier contact (999 = jamais contacté)",
                    min_value=0,
                    max_value=999,
                    value=999
                )

                previous = st.number_input(
                    "Nombre de contacts avant cette campagne",
                    min_value=0,
                    max_value=100,
                    value=0
                )

                poutcome = st.selectbox(
                    "Résultat de la campagne précédente",
                    encoders["poutcome"].classes_.tolist()
                )

            submitted = st.form_submit_button(
                "Prédire la souscription"
            )

        if submitted:

            # Encodage des variables catégorielles
            categorical_values = {
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "month": month,
                "day_of_week": day_of_week,
                "poutcome": poutcome
            }

            encoded_values = {
                col: int(encoders[col].transform([value])[0])
                for col, value in categorical_values.items()
            }

            # Respecter exactement l'ordre des variables
            input_data = np.array([[
                age,
                encoded_values["job"],
                encoded_values["marital"],
                encoded_values["education"],
                encoded_values["default"],
                encoded_values["housing"],
                encoded_values["loan"],
                encoded_values["contact"],
                encoded_values["month"],
                encoded_values["day_of_week"],
                duration,
                campaign,
                pdays,
                previous,
                encoded_values["poutcome"]
            ]], dtype=np.float64)

            # Normalisation
            input_scaled = scaler.transform(input_data)

            # Prédiction
            model = load_model("DNN_optimise.keras")
            probability = float(
                model.predict(input_scaled, verbose=0)[0][0]
            )

            st.subheader("Résultat de la prédiction")

            if probability >= 0.5:

                st.success(
                    "Le modèle prédit que le client est susceptible "
                    "de souscrire."
                )

            else:

                st.warning(
                    "Le modèle prédit que le client n'est pas "
                    "susceptible de souscrire."
                )

            st.metric(
                "Probabilité estimée de souscription",
                f"{probability * 100:.2f} %"
            )

            st.progress(
                float(np.clip(probability, 0, 1))
            )

    except FileNotFoundError as e:

        st.error(
            f"Fichier nécessaire introuvable : {e.filename}"
        )

    except Exception as e:

        st.error(f"Erreur lors de la prédiction : {e}")


# =========================
# STUDENT PERFORMANCE
# =========================


elif projet == "Student Performance":

    st.header("Prédiction des performances scolaires")

    st.write(
        "Estimation de l'indice de performance d'un étudiant "
        "à partir de ses habitudes d'étude."
    )

    try:
        encoder, scaler = load_student_preprocessing()

        with st.form("student_form"):

            st.subheader("Informations de l'étudiant")

            hours = st.number_input(
                "Heures d'étude",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5
            )

            previous_scores = st.number_input(
                "Notes précédentes",
                min_value=0.0,
                max_value=100.0,
                value=70.0
            )

            activities = st.selectbox(
                "Activités extrascolaires",
                ["Yes", "No"]
            )

            sleep = st.number_input(
                "Heures de sommeil",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.5
            )

            papers = st.number_input(
                "Sujets d'examen pratiqués",
                min_value=0,
                max_value=100,
                value=4
            )

            submitted = st.form_submit_button(
                "Prédire la performance"
            )

        if submitted:

            activity_encoded = int(
                encoder.transform([activities])[0]
            )

            input_data = np.array([[
                hours,
                previous_scores,
                activity_encoded,
                sleep,
                papers
            ]], dtype=np.float64)

            input_scaled = scaler.transform(input_data)

            model = load_model(
                "DNN_Student_Performance.keras"
            )

            prediction = float(
                model.predict(input_scaled, verbose=0)[0][0]
            )

            st.subheader("Résultat de la prédiction")

            st.metric(
                "Performance Index estimé",
                f"{prediction:.2f}"
            )

            st.info(
                "Cette valeur est une estimation produite "
                "par le modèle, et non une note garantie."
            )

    except FileNotFoundError as e:

        st.error(f"Fichier introuvable : {e.filename}")

    except Exception as e:

        st.error(f"Erreur lors de la prédiction : {e}")


# =========================
# CIFAR-10
# =========================

elif projet == "CIFAR-10":

    st.header("Classification d'images CIFAR-10")

    classes = [
        "Avion", "Automobile", "Oiseau", "Chat", "Cerf",
        "Chien", "Grenouille", "Cheval", "Bateau", "Camion"
    ]

    uploaded = st.file_uploader(
        "Importer une image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded)

        st.image(
            image,
            caption="Image importée",
            width=350
        )

        if st.button("Classifier l'image"):

            with st.spinner("Classification en cours..."):

                model = load_model("CNN_CIFAR10_optimise.keras")

                predictions = predict_image(
                    model,
                    image,
                    (32, 32)
                )

                index = int(np.argmax(predictions))

            st.success(
                f"Classe prédite : **{classes[index]}**"
            )

            st.write(
                f"Confiance : {predictions[index] * 100:.2f} %"
            )

            st.subheader("Probabilités par classe")

            for i, classe in enumerate(classes):

                st.write(
                    f"{classe} : {predictions[i] * 100:.2f} %"
                )


# =========================
# PNEUMONIA
# =========================

elif projet == "Pneumonia":

    st.header("🫁 Détection de pneumonie")

    st.warning(
        "Prototype pédagogique uniquement, "
        "pas un outil de diagnostic médical."
    )

    uploaded = st.file_uploader(
        "Importer une radiographie pulmonaire",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded)

        st.image(
            image,
            caption="Image importée",
            width=350
        )

        if st.button("Analyser la radiographie"):

            with st.spinner("Analyse en cours..."):

                model = load_model(
                    "modele_pneumonia_final.keras"
                )

                probability = float(
                    predict_image(
                        model,
                        image,
                        (150, 150)
                    )[0]
                )

            st.subheader("Résultat")

            if probability >= 0.5:

                st.error(
                    f"PNEUMONIA : {probability * 100:.2f} %"
                )

                st.write(
                    f"NORMAL : {(1-probability)*100:.2f} %"
                )

            else:

                st.success(
                    f"NORMAL : {(1-probability)*100:.2f} %"
                )

                st.write(
                    f"PNEUMONIA : {probability*100:.2f} %"
                )


# =========================
# TUBERCULOSIS
# =========================

elif projet == "Tuberculosis":

    st.header("🫁 Détection de tuberculose")

    st.warning(
        "Prototype pédagogique uniquement, "
        "pas un outil de diagnostic médical."
    )

    uploaded = st.file_uploader(
        "Importer une radiographie pulmonaire",
        type=["jpg", "jpeg", "png"],
        key="tb_upload"
    )

    if uploaded:

        image = Image.open(uploaded)

        st.image(
            image,
            caption="Image importée",
            width=350
        )

        if st.button("Analyser la radiographie"):

            with st.spinner("Analyse en cours..."):

                model = load_tb_model()

                probability = float(
                    predict_image(
                        model,
                        image,
                        (150, 150)
                    )[0]
                )

            st.subheader("Résultat")

            if probability >= 0.5:

                st.error(
                    f"TUBERCULOSIS : {probability * 100:.2f} %"
                )

                st.write(
                    f"NORMAL : {(1-probability)*100:.2f} %"
                )

            else:

                st.success(
                    f"NORMAL : {(1-probability)*100:.2f} %"
                )

                st.write(
                    f"TUBERCULOSIS : {probability*100:.2f} %"
                )
