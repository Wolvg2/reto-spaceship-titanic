"""
Spaceship Titanic - Equipo 5
Interfaz local para probar el modelo con un pasajero hipotético.

Cómo correrla:
    pip install -r requirements.txt
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Spaceship Titanic - Predicción")


# Carga del modelo y los artifacts de preprocesamiento (ajustados en el
# notebook con train.csv
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_final.joblib")
    artifacts = joblib.load("artifacts.joblib")
    return modelo, artifacts


modelo, art = cargar_modelo()


def preprocesar_pasajero(raw: dict) -> pd.DataFrame:
    """Replica exactamente el pipeline del primer avance para un solo pasajero."""

    df = pd.DataFrame([raw])

    # --- Imputación ---
    if pd.isna(df.at[0, "CryoSleep"]):
        df.at[0, "CryoSleep"] = art["cryo_mode"]
    if pd.isna(df.at[0, "VIP"]):
        df.at[0, "VIP"] = art["vip_mode"]

    home_planet = df.at[0, "HomePlanet"]

    if pd.isna(df.at[0, "Age"]):
        df.at[0, "Age"] = art["age_medians_by_planet"].get(
            home_planet, art["age_median_global"]
        )

    # Regla CryoSleep=True -> gasto 0, igual que en el primer avance
    for col in art["spending_cols"]:
        if df.at[0, "CryoSleep"] and pd.isna(df.at[0, col]):
            df.at[0, col] = 0
        elif pd.isna(df.at[0, col]):
            df.at[0, col] = art["spending_medians_by_planet"][col].get(
                home_planet, art["spending_medians_global"][col]
            )

    for col in ["HomePlanet", "Deck", "Side", "Destination"]:
        if pd.isna(df.at[0, col]):
            df.at[0, col] = "Unknown"

    # --- Codificación ---
    df["CryoSleep"] = df["CryoSleep"].astype(int)
    df["VIP"] = df["VIP"].astype(int)
    df["Deck"] = df["Deck"].map(art["deck_order"])

    nominal_cols = [c for c in ["HomePlanet", "Destination", "Side"] if c in df.columns]
    df = pd.get_dummies(df, columns=nominal_cols, dtype=int)
    df = df.reindex(columns=art["dummy_columns"], fill_value=0)

    # --- log1p en las variables sesgadas ---
    for col in art["high_skewed_features"]:
        df[col] = np.log1p(df[col])

    # --- Escalado  ---
    df[art["numerical_cols_to_scale"]] = art["scaler"].transform(
        df[art["numerical_cols_to_scale"]]
    )

    return df



# Formulario
st.title(" Spaceship Titanic — Predecir")
st.caption(
 "Modelo: Random Forest tuneado con `GridSearchCV` (`n_estimators=300`, `max_depth=14`, "
 "`min_samples_leaf=8`, `max_features=0.5`)"
)

col1, col2 = st.columns(2)

with col1:
    home_planet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
    destination = st.selectbox(
        "Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]
    )
    deck = st.selectbox("Deck (cubierta)", ["A", "B", "C", "D", "E", "F", "G", "T"])
    side = st.selectbox("Side (lado del pasillo)", ["P", "S"])
    cryo_sleep = st.checkbox("¿Está en CryoSleep?")
    vip = st.checkbox("¿Es VIP?")

with col2:
    age = st.number_input("Edad", min_value=0, max_value=100, value=27)
    solo = st.checkbox("¿Viaja solo?", value=True)
    group_size = 1 if solo else st.number_input(
        "Tamaño de su grupo", min_value=2, max_value=10, value=2
    )
    room_service = st.number_input("Gasto en RoomService", min_value=0.0, value=0.0)
    food_court = st.number_input("Gasto en FoodCourt", min_value=0.0, value=0.0)
    shopping_mall = st.number_input("Gasto en ShoppingMall", min_value=0.0, value=0.0)
    spa = st.number_input("Gasto en Spa", min_value=0.0, value=0.0)
    vr_deck = st.number_input("Gasto en VRDeck", min_value=0.0, value=0.0)

if st.button("Predecir", type="primary"):
    raw = {
        "GroupSize": group_size,
        "TravelAlone": int(solo),
        "HomePlanet": home_planet,
        "CryoSleep": cryo_sleep,
        "Deck": deck,
        "Side": side,
        "Destination": destination,
        "Age": age,
        "VIP": vip,
        "RoomService": room_service,
        "FoodCourt": food_court,
        "ShoppingMall": shopping_mall,
        "Spa": spa,
        "VRDeck": vr_deck,
    }

    X_nuevo = preprocesar_pasajero(raw)
    pred = modelo.predict(X_nuevo)[0]
    proba = modelo.predict_proba(X_nuevo)[0, 1]

    st.divider()
    if pred == 1:
        st.success(f" Predicción: **Transportado** (probabilidad: {proba:.1%})")
    else:
        st.error(f" Predicción: **No transportado** (probabilidad de sí: {proba:.1%})")

    st.progress(float(proba))

    with st.expander("Ver features procesadas que recibió el modelo"):
        st.dataframe(X_nuevo)
