import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DeepLearning Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = Path(__file__).parent / "models"

# =========================================================
# DESIGN SYSTEM — Palette moderne & épurée
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700;800&display=swap');

    :root {
        /* Couleurs */
        --bg: #FAFAFB;
        --surface: #FFFFFF;
        --surface-2: #F5F6F8;
        --ink: #0A0E1A;
        --ink-2: #2A3142;
        --muted: #6B7280;
        --muted-2: #9CA3AF;
        --line: #ECEEF2;
        --line-2: #E2E5EB;

        --primary: #6366F1;        /* Indigo moderne */
        --primary-dark: #4F46E5;
        --primary-light: #EEF0FF;
        --accent: #06B6D4;         /* Cyan */
        --accent-dark: #0891B2;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;

        /* Radius */
        --r-sm: 10px;
        --r-md: 14px;
        --r-lg: 18px;
        --r-xl: 24px;

        /* Shadows */
        --shadow-xs: 0 1px 2px rgba(10, 14, 26, .04);
        --shadow-sm: 0 2px 8px rgba(10, 14, 26, .05);
        --shadow-md: 0 6px 20px rgba(10, 14, 26, .07);
        --shadow-lg: 0 16px 40px rgba(10, 14, 26, .10);
        --shadow-primary: 0 8px 24px rgba(99, 102, 241, .25);
    }

    /* ============ BASE ============ */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--ink);
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background: var(--bg);
    }

    [data-testid="stHeader"] {
        background: rgba(250, 250, 251, 0.85);
        backdrop-filter: blur(8px);
    }

    .block-container {
        max-width: 1240px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        font-family: 'Sora', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -.03em;
    }
    h1 { font-weight: 800 !important; }
    h2, h3 { font-weight: 700 !important; }

    /* ============ SIDEBAR ============ */
    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }

    /* Brand */
    .brand-mark {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0 0 1.5rem 0;
    }

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 14px;
        color: white;
        font-size: 20px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        box-shadow: var(--shadow-primary);
    }

    .brand-name {
        color: var(--ink);
        font-family: 'Sora', sans-serif;
        font-size: 16px;
        line-height: 1.15;
        font-weight: 700;
        letter-spacing: -.02em;
    }

    .brand-sub {
        color: var(--muted-2);
        font-size: 10px;
        margin-top: 4px;
        letter-spacing: .12em;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* Nav custom */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 4px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 10px 14px !important;
        border-radius: var(--r-sm);
        transition: all .18s ease;
        font-size: 13.5px;
        font-weight: 500;
        color: var(--ink-2) !important;
        border: 1px solid transparent;
        cursor: pointer;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: var(--surface-2);
    }

    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: var(--primary-light);
        color: var(--primary-dark) !important;
        border-color: rgba(99, 102, 241, .2);
        font-weight: 600;
    }

    /* ============ TYPOGRAPHIE UI ============ */
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--primary);
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .12em;
        margin-bottom: 12px;
        padding: 6px 12px;
        background: var(--primary-light);
        border-radius: 999px;
    }

    .page-title {
        color: var(--ink);
        font-family: 'Sora', sans-serif;
        font-size: clamp(28px, 3.2vw, 40px);
        font-weight: 800;
        line-height: 1.12;
        letter-spacing: -.045em;
        margin: 0 0 10px 0;
    }

    .page-subtitle {
        color: var(--muted);
        font-size: 15.5px;
        line-height: 1.65;
        margin-bottom: 1.8rem;
        max-width: 760px;
    }

    /* ============ HERO ============ */
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 45%, #4F46E5 100%);
        border-radius: var(--r-xl);
        padding: 40px 42px;
        color: #FFFFFF;
        margin: 0 0 30px 0;
        box-shadow: var(--shadow-lg);
    }

    .hero::before {
        content: '';
        position: absolute;
        top: -60%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(6, 182, 212, .3) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: 30%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99, 102, 241, .25) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-kicker {
        position: relative;
        color: #C7D2FE;
        text-transform: uppercase;
        letter-spacing: .14em;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 14px;
    }

    .hero-title {
        position: relative;
        color: #FFFFFF;
        font-family: 'Sora', sans-serif;
        font-size: clamp(30px, 3.6vw, 44px);
        font-weight: 800;
        letter-spacing: -.04em;
        line-height: 1.1;
        max-width: 720px;
        margin-bottom: 14px;
    }

    .hero-copy {
        position: relative;
        color: #E0E7FF;
        font-size: 15.5px;
        line-height: 1.7;
        max-width: 680px;
    }

    .hero-chip {
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid rgba(255,255,255,.18);
        background: rgba(255,255,255,.08);
        backdrop-filter: blur(6px);
        border-radius: 999px;
        padding: 8px 16px;
        color: #F3F4F6;
        font-size: 12.5px;
        font-weight: 500;
        margin-top: 22px;
    }

    /* ============ SECTIONS ============ */
    .section-heading {
        font-family: 'Sora', sans-serif;
        color: var(--ink);
        font-size: 21px;
        font-weight: 700;
        letter-spacing: -.025em;
        margin: 32px 0 6px 0;
    }

    .section-caption {
        color: var(--muted);
        font-size: 13.5px;
        margin-bottom: 18px;
    }

    /* ============ PROJECT CARDS ============ */
    .project-card {
        position: relative;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--r-lg);
        padding: 24px;
        min-height: 190px;
        box-shadow: var(--shadow-xs);
        transition: all .22s ease;
        overflow: hidden;
    }

    .project-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        opacity: 0;
        transition: opacity .22s ease;
    }

    .project-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
        border-color: var(--line-2);
    }

    .project-card:hover::before {
        opacity: 1;
    }

    .project-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: var(--primary-light);
        font-size: 24px;
        margin-bottom: 16px;
        transition: transform .22s ease;
    }

    .project-card:hover .project-icon {
        transform: scale(1.05);
    }

    .project-title {
        color: var(--ink);
        font-family: 'Sora', sans-serif;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: -.01em;
    }

    .project-desc {
        color: var(--muted);
        font-size: 12.5px;
        line-height: 1.6;
    }

    /* ============ INFO STRIP ============ */
    .info-strip {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        border: 1px solid #DBEAFE;
        background: #EFF6FF;
        border-radius: var(--r-md);
        padding: 16px 20px;
        color: #1E40AF;
        font-size: 13.5px;
        line-height: 1.6;
        margin: 16px 0 24px 0;
    }

    .info-strip.warning {
        border-color: #FED7AA;
        background: #FFF7ED;
        color: #9A3412;
    }

    /* ============ RESULT PANELS ============ */
    .result-panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--r-lg);
        padding: 26px;
        margin-top: 20px;
        box-shadow: var(--shadow-sm);
    }

    .result-label {
        color: var(--muted-2);
        text-transform: uppercase;
        letter-spacing: .1em;
        font-size: 10.5px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .result-value {
        color: var(--ink);
        font-family: 'Sora', sans-serif;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -.03em;
    }

    /* ============ FOOTER ============ */
    .footer {
        border-top: 1px solid var(--line);
        margin-top: 56px;
        padding-top: 22px;
        color: var(--muted-2);
        font-size: 11.5px;
        text-align: center;
        line-height: 1.7;
    }

    /* ============ FORMULAIRES ============ */
    div[data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--r-lg);
        padding: 26px 26px 14px 26px;
        box-shadow: var(--shadow-xs);
    }

    div[data-testid="stForm"] h3 {
        margin-top: 0;
        margin-bottom: 18px;
    }

    /* ============ BOUTONS ============ */
    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: var(--primary) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--primary) !important;
        border-radius: var(--r-sm) !important;
        padding: .68rem 1.4rem !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all .18s ease !important;
        box-shadow: var(--shadow-xs);
    }

    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: var(--primary-dark) !important;
        border-color: var(--primary-dark) !important;
        box-shadow: var(--shadow-primary);
        transform: translateY(-1px);
    }

    /* ============ INPUTS ============ */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stFileUploader"] label {
        color: var(--ink-2);
        font-weight: 600;
        font-size: 13px;
    }

    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stNumberInput"] input {
        border-radius: var(--r-sm) !important;
        border-color: var(--line-2) !important;
    }

    div[data-testid="stSelectbox"] > div > div:focus-within,
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, .12) !important;
    }

    /* ============ FILE UPLOADER ============ */
    div[data-testid="stFileUploader"] section {
        background: var(--surface-2);
        border: 1.5px dashed var(--line-2);
        border-radius: var(--r-md);
        transition: all .18s ease;
    }

    div[data-testid="stFileUploader"] section:hover {
        border-color: var(--primary);
        background: var(--primary-light);
    }

    /* ============ METRICS ============ */
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--r-md);
        padding: 18px 20px;
        box-shadow: var(--shadow-xs);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 700 !important;
    }

    /* ============ ALERTS ============ */
    [data-testid="stAlert"] {
        border-radius: var(--r-md);
        border: 1px solid var(--line);
    }

    /* ============ PROGRESS ============ */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
    }

    /* ============ ANIMATIONS ============ */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .hero, .project-card, .result-panel {
        animation: fadeInUp .45s ease-out;
    }

    /* ============ CLEANUP ============ */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tabs Streamlit custom */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: var(--surface-2);
        padding: 6px;
        border-radius: var(--r-md);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: var(--r-sm);
        padding: 8px 16px;
        font-weight: 600;
        font-size: 13px;
    }

    .stTabs [aria-selected="true"] {
        background: var(--surface) !important;
        color: var(--primary-dark) !important;
        box-shadow: var(--shadow-xs);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CHARGEMENT DES MODÈLES
# =========================================================

@st.cache_resource
def load_model(filename):
    return tf.keras.models.load_model(MODEL_DIR / filename, compile=False)


@st.cache_resource
def load_tb_model():
    model_path = hf_hub_download(
        repo_id="DouceR/tuberculosis-model",
        filename="vgg16_model_finetuned.h5"
    )
    return tf.keras.models.load_model(model_path, compile=False)


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


# =========================================================
# PRÉTRAITEMENT
# =========================================================

def predict_image(model, image, size):
    image = image.convert("RGB").resize(size)
    array = np.array(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    return model.predict(array, verbose=0)[0]


# =========================================================
# COMPOSANTS UI
# =========================================================

def page_heading(eyebrow, title, subtitle):
    st.markdown(
        f"""
        <div class="eyebrow">{eyebrow}</div>
        <div class="page-title">{title}</div>
        <div class="page-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True
    )


def project_card(icon, title, description):
    st.markdown(
        f"""
        <div class="project-card">
            <div class="project-icon">{icon}</div>
            <div class="project-title">{title}</div>
            <div class="project-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_image_upload(label, key=None):
    uploaded = st.file_uploader(label, type=["jpg", "jpeg", "png"], key=key)
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([1, 1.35], gap="large")
        with left:
            st.image(image, caption="Aperçu de l'image importée", use_container_width=True)
        with right:
            st.markdown("**Image prête pour l'analyse**")
            st.caption(f"Dimensions originales : {image.width} × {image.height} pixels")
            st.caption("Formats acceptés : JPG, JPEG et PNG")
        return image
    return None


def show_medical_notice():
    st.markdown(
        """
        <div class="info-strip warning">
            <div>
                <strong>Information importante :</strong> ce module est un prototype
                pédagogique destiné à illustrer la classification d'images.
                Il ne remplace pas un examen médical ni l'avis d'un professionnel de santé.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_binary_medical_result(probability, positive_label):
    negative_label = "NORMAL"
    positive_percent = probability * 100
    negative_percent = (1 - probability) * 100

    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    st.markdown('<div class="result-label">Résultat du modèle</div>', unsafe_allow_html=True)

    if probability >= 0.5:
        st.error(f"Classe prédite : **{positive_label}**")
    else:
        st.success(f"Classe prédite : **{negative_label}**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Score {positive_label}", f"{positive_percent:.2f} %")
    with col2:
        st.metric(f"Score {negative_label}", f"{negative_percent:.2f} %")

    st.progress(float(np.clip(probability, 0, 1)), text=f"Score associé à {positive_label}")
    st.caption("Ces scores reflètent la sortie du modèle et ne constituent pas un diagnostic.")
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# NAVIGATION
# =========================================================

PROJECTS = [
    "Accueil",
    "Bank Marketing",
    "Student Performance",
    "CIFAR-10",
    "Pneumonia",
    "Tuberculosis"
]

st.sidebar.markdown(
    """
    <div class="brand-mark">
        <div class="brand-icon">✦</div>
        <div>
            <div class="brand-name">DeepLearning Studio</div>
            <div class="brand-sub">AI · Machine Learning</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:10px;letter-spacing:.12em;color:#9CA3AF;font-weight:700;margin-bottom:8px;">NAVIGATION</div>',
    unsafe_allow_html=True
)

projet = st.sidebar.radio("Navigation", PROJECTS, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size:12px;color:#6B7280;line-height:1.7;">
        <strong style="color:#0A0E1A;">Plateforme académique</strong><br>
        Démonstration de cinq modèles de Deep Learning.<br><br>
        <span style="color:#10B981;">●</span> Modules disponibles
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ACCUEIL
# =========================================================

if projet == "Accueil":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Plateforme de démonstration · Intelligence artificielle</div>
            <div class="hero-title">Explorez le potentiel du Deep Learning.</div>
            <div class="hero-copy">
                Une interface unique pour découvrir, tester et comprendre
                cinq applications de réseaux de neurones sur des données
                structurées et des images.
            </div>
            <div class="hero-chip">✦ 5 projets · 5 modèles · 1 plateforme</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page_heading(
        "VOTRE ESPACE IA",
        "Bienvenue sur DeepLearning Studio",
        "Sélectionnez un projet dans le menu latéral pour accéder à son interface, "
        "renseigner les données ou importer une image et consulter la prédiction."
    )

    st.markdown('<div class="section-heading">Vos projets</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Cinq cas d\'usage, une même expérience simple et claire.</div>',
        unsafe_allow_html=True
    )

    cards = [
        ("📊", "Bank Marketing", "Estimer la probabilité qu'un client souscrive à un dépôt bancaire."),
        ("🎓", "Student Performance", "Estimer un indice de performance à partir des habitudes d'étude."),
        ("🖼️", "CIFAR-10", "Classer une image parmi dix catégories d'objets."),
        ("🫁", "Pneumonia", "Explorer un modèle de classification de radiographies pulmonaires."),
        ("🔬", "Tuberculosis", "Explorer un modèle de classification de radiographies pulmonaires.")
    ]

    for start in range(0, len(cards), 3):
        cols = st.columns(3, gap="medium")
        for col, card in zip(cols, cards[start:start + 3]):
            with col:
                project_card(*card)

    st.markdown('<div class="section-heading">Comment utiliser la plateforme ?</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-strip">
            <div>
                <strong>1.</strong> Choisissez un module dans le menu à gauche.
                &nbsp;&nbsp; <strong>2.</strong> Saisissez les informations ou importez une image.
                &nbsp;&nbsp; <strong>3.</strong> Lancez la prédiction et consultez le résultat.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# BANK MARKETING
# =========================================================

elif projet == "Bank Marketing":
    page_heading(
        "MODÈLE 01 · DONNÉES BANCAIRES",
        "Prédiction de souscription bancaire",
        "Estimez si un client est susceptible de souscrire à un dépôt bancaire "
        "à partir de ses caractéristiques et de l'historique de contact."
    )

    st.markdown(
        '<div class="info-strip">Renseignez les informations du client. '
        'Les champs sont organisés en deux groupes pour faciliter la saisie.</div>',
        unsafe_allow_html=True
    )

    try:
        encoders, scaler = load_bank_preprocessing()

        with st.form("bank_form"):
            st.subheader("Profil du client")
            col1, col2 = st.columns(2, gap="large")

            with col1:
                age = st.number_input("Âge", min_value=18, max_value=100, value=35)
                job = st.selectbox("Profession", encoders["job"].classes_.tolist())
                marital = st.selectbox("Situation matrimoniale", encoders["marital"].classes_.tolist())
                education = st.selectbox("Niveau d'éducation", encoders["education"].classes_.tolist())
                default = st.selectbox("Crédit en défaut ?", encoders["default"].classes_.tolist())
                housing = st.selectbox("Crédit immobilier ?", encoders["housing"].classes_.tolist())
                loan = st.selectbox("Crédit personnel ?", encoders["loan"].classes_.tolist())
                contact = st.selectbox("Type de contact", encoders["contact"].classes_.tolist())

            with col2:
                st.markdown("**Historique de contact**")
                month = st.selectbox("Mois du dernier contact", encoders["month"].classes_.tolist())
                day_of_week = st.selectbox("Jour de la semaine", encoders["day_of_week"].classes_.tolist())
                duration = st.number_input("Durée du dernier contact (secondes)", min_value=0, max_value=10000, value=180)
                campaign = st.number_input("Nombre de contacts pendant cette campagne", min_value=1, max_value=100, value=1)
                pdays = st.number_input("Jours depuis le dernier contact (999 = jamais contacté)", min_value=0, max_value=999, value=999)
                previous = st.number_input("Nombre de contacts avant cette campagne", min_value=0, max_value=100, value=0)
                poutcome = st.selectbox("Résultat de la campagne précédente", encoders["poutcome"].classes_.tolist())

            submitted = st.form_submit_button("Lancer la prédiction", use_container_width=True)

        if submitted:
            categorical_values = {
                "job": job, "marital": marital, "education": education,
                "default": default, "housing": housing, "loan": loan,
                "contact": contact, "month": month,
                "day_of_week": day_of_week, "poutcome": poutcome
            }

            encoded_values = {
                col: int(encoders[col].transform([value])[0])
                for col, value in categorical_values.items()
            }

            input_data = np.array([[
                age, encoded_values["job"], encoded_values["marital"],
                encoded_values["education"], encoded_values["default"],
                encoded_values["housing"], encoded_values["loan"],
                encoded_values["contact"], encoded_values["month"],
                encoded_values["day_of_week"], duration, campaign, pdays,
                previous, encoded_values["poutcome"]
            ]], dtype=np.float64)

            input_scaled = scaler.transform(input_data)
            model = load_model("DNN_optimise.keras")
            probability = float(model.predict(input_scaled, verbose=0)[0][0])

            st.markdown("### Résultat de l'estimation")
            col1, col2 = st.columns([1.2, 1], gap="large")
            with col1:
                if probability >= 0.5:
                    st.success("Le modèle estime que le client est susceptible de souscrire.")
                else:
                    st.info("Le modèle estime que le client est peu susceptible de souscrire.")
                st.progress(float(np.clip(probability, 0, 1)), text="Probabilité estimée de souscription")
            with col2:
                st.metric("Probabilité estimée", f"{probability * 100:.2f} %")
            st.caption("Résultat indicatif produit par le modèle à partir des informations saisies.")

    except FileNotFoundError as e:
        st.error(f"Fichier nécessaire introuvable : {e.filename}")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")


# =========================================================
# STUDENT PERFORMANCE
# =========================================================

elif projet == "Student Performance":
    page_heading(
        "MODÈLE 02 · ÉDUCATION",
        "Estimation de la performance scolaire",
        "Obtenez une estimation de l'indice de performance à partir des habitudes "
        "d'étude, des résultats précédents et du rythme de vie de l'étudiant."
    )

    try:
        encoder, scaler = load_student_preprocessing()

        with st.form("student_form"):
            st.subheader("Informations de l'étudiant")
            col1, col2 = st.columns(2, gap="large")

            with col1:
                hours = st.number_input("Heures d'étude par jour", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
                previous_scores = st.number_input("Notes précédentes", min_value=0.0, max_value=100.0, value=70.0)
                activities = st.selectbox("Activités extrascolaires", ["Yes", "No"])

            with col2:
                sleep = st.number_input("Heures de sommeil par jour", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
                papers = st.number_input("Sujets d'examen pratiqués", min_value=0, max_value=100, value=4)

            submitted = st.form_submit_button("Estimer la performance", use_container_width=True)

        if submitted:
            activity_encoded = int(encoder.transform([activities])[0])
            input_data = np.array([[hours, previous_scores, activity_encoded, sleep, papers]], dtype=np.float64)

            input_scaled = scaler.transform(input_data)
            model = load_model("DNN_Student_Performance.keras")
            prediction = float(model.predict(input_scaled, verbose=0)[0][0])

            st.markdown("### Résultat de l'estimation")
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.metric("Performance Index estimé", f"{prediction:.2f}")
            st.caption("Cette valeur est une estimation produite par le modèle, et non une note garantie.")
            st.markdown('</div>', unsafe_allow_html=True)

    except FileNotFoundError as e:
        st.error(f"Fichier introuvable : {e.filename}")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")


# =========================================================
# CIFAR-10
# =========================================================

elif projet == "CIFAR-10":
    page_heading(
        "MODÈLE 03 · VISION PAR ORDINATEUR",
        "Classification d'images CIFAR-10",
        "Importez une image pour que le réseau de neurones convolutif "
        "la classe parmi dix catégories d'objets."
    )

    classes = [
        "Avion", "Automobile", "Oiseau", "Chat", "Cerf",
        "Chien", "Grenouille", "Cheval", "Bateau", "Camion"
    ]

    image = show_image_upload("Importer une image à classifier")

    if image and st.button("Classifier l'image", use_container_width=True):
        with st.spinner("Classification en cours…"):
            model = load_model("CNN_CIFAR10_optimise.keras")
            predictions = predict_image(model, image, (32, 32))
            index = int(np.argmax(predictions))

        st.markdown("### Résultat de la classification")
        left, right = st.columns([1, 1.3], gap="large")
        with left:
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown('<div class="result-label">Classe prédite</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-value">{classes[index]}</div>', unsafe_allow_html=True)
            st.metric("Confiance du modèle", f"{predictions[index] * 100:.2f} %")
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown("**Probabilités par catégorie**")
            for i, classe in enumerate(classes):
                st.write(f"{classe} · {predictions[i] * 100:.2f} %")
                st.progress(float(np.clip(predictions[i], 0, 1)))


# =========================================================
# PNEUMONIA
# =========================================================

elif projet == "Pneumonia":
    page_heading(
        "MODÈLE 04 · CLASSIFICATION MÉDICALE EXPÉRIMENTALE",
        "Classification de radiographies · Pneumonia",
        "Importez une radiographie pulmonaire pour consulter la sortie du modèle "
        "de classification entraîné dans le cadre du projet."
    )
    show_medical_notice()

    image = show_image_upload("Importer une radiographie pulmonaire")

    if image and st.button("Analyser la radiographie", use_container_width=True):
        with st.spinner("Analyse en cours…"):
            model = load_model("modele_pneumonia_final.keras")
            probability = float(predict_image(model, image, (150, 150))[0])

        st.markdown("### Résultat de la classification")
        show_binary_medical_result(probability, "PNEUMONIA")


# =========================================================
# TUBERCULOSIS
# =========================================================

elif projet == "Tuberculosis":
    page_heading(
        "MODÈLE 05 · CLASSIFICATION MÉDICALE EXPÉRIMENTALE",
        "Classification de radiographies · Tuberculosis",
        "Importez une radiographie pulmonaire pour consulter la sortie du modèle "
        "de classification hébergé sur Hugging Face."
    )
    show_medical_notice()

    image = show_image_upload("Importer une radiographie pulmonaire", key="tb_upload")

    if image and st.button("Analyser la radiographie", use_container_width=True):
        with st.spinner("Chargement du modèle et analyse en cours…"):
            model = load_tb_model()
            probability = float(predict_image(model, image, (150, 150))[0])

        st.markdown("### Résultat de la classification")
        show_binary_medical_result(probability, "TUBERCULOSIS")


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        DeepLearning Studio · Plateforme académique de démonstration
        <br>Les prédictions sont expérimentales et doivent être interprétées avec discernement.
    </div>
    """,
    unsafe_allow_html=True
)