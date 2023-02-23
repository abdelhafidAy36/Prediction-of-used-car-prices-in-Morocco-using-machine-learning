import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib
import pickle
from ast import literal_eval
from pycaret.regression import *

# encoding
mlb = pickle.load(open('files/mlb.pkl', 'rb'))

# model
model = joblib.load('files/catboost_model.pkl')


@st.cache(show_spinner=False)
def get_row_data():
    data = pd.read_csv("data/cars_data.csv", encoding='latin1')
    return data


def clean_data(data):
    df = data.copy()
    # Nom -> (Marque + Model)
    Marque = df['Nom'].str.split(' ').str[0]
    df.insert(1, "Marque", Marque)
    Model = df['Nom'].str.split(' ').str[1:].str.join(" ")
    df.insert(2, "Model", Model)
    df.drop(['Nom'], axis=1, inplace=True)

    # les formats incorrects
    df["Kilométrage"] = df["Kilométrage"].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(
        int)
    df["Prix"] = df["Prix"].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int)
    df["Première main"].fillna("Non", inplace=True)
    df["Voiture personnalisée (tuning)"].fillna("Non", inplace=True)
    df["Importé neuf"].fillna("Non", inplace=True)
    df["Véhicule dédouané"] = df["Véhicule dédouané"].apply(lambda x: "Oui" if pd.notna(x) else "Non")
    df["Véhicule en garantie"] = df["Véhicule en garantie"].apply(lambda x: "Oui" if pd.notna(x) else "Non")

    # Supprimer les colonnes inutiles et les doublons
    df.drop("Date", axis=1, inplace=True)
    df.drop_duplicates(inplace=True)

    # misssing values
    # Kilométrage
    df['Kilométrage'].fillna(df.groupby('Année')['Kilométrage'].transform('mean'), inplace=True)
    df['Kilométrage'].fillna(df['Kilométrage'].mean(), inplace=True)

    # Couleur
    df['Couleur'].fillna(method="ffill", inplace=True)

    # Puissance fiscale et Nombre de portes
    num_cols = ["Puissance fiscale", "Nombre de portes"]

    for col in num_cols:
        df[col].fillna(df.groupby(['Marque', 'Model'])[col].transform('mean'), inplace=True)
        df[col].fillna(df[col].mean(), inplace=True)

    # Carrosserie, Boite de vitesses et Options
    cat_cols = ["Carrosserie", "Boite de vitesses", "Options"]

    for col in cat_cols:
        df[col].fillna(df.groupby(['Marque', 'Model'])[col].apply(lambda x: x.ffill().bfill()), inplace=True)
        df[col].fillna(method="ffill", inplace=True)

    # outliers
    # kil
    def find_outliers_limit(df, col):
        # removing outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # calculate the outlier cutoff
        cut_off = IQR * 1.5
        lower, upper = Q1 - cut_off, Q3 + cut_off

        return lower, upper

    def remove_outlier(df, col, upper, lower):
        # remove outliers
        df = df.loc[(df[col] > lower) & (df[col] < upper)]

        return df

    lower, upper = find_outliers_limit(df, "Kilométrage")
    df = remove_outlier(df, "Kilométrage", upper, lower)
    # prix
    df = df.loc[(df["Prix"] < 1000000)]

    # eda
    df = df.loc[df["Carburant"] != "Hybride"]
    df = df.loc[df["Carburant"] != "Electrique"]
    df.drop("Voiture personnalisée (tuning)", axis=1, inplace=True)
    df.drop("Nombre de portes", axis=1, inplace=True)

    return df


def expl():
    st.title("📈 DRIFT DES DONNÉES")

    st.markdown("")
    st.markdown("")

    # curr data
    col1, col2 = st.columns([8, 4])

    with col1:
        st.markdown(
            "Le Drift rapport des données permet de détecter et d'explorer les changements dans les données d'entrée.")
        st.markdown(
            "**Exigences :** Vous aurez besoin de deux ensembles de données. L'ensemble de données de référence (les données utilisées pour entraîner le modèle final) et l'ensemble de données actuelles (les données de production). Pour plus d'informations sur Data Drift, consultez ce document : [link](https://docs.evidentlyai.com/reports/data-drift)")

    with col2:
        uploaded_file = st.file_uploader("Définir les données courantes :", accept_multiple_files=False)

    drift_report = st.empty()

    with drift_report.container():
        html_file = open("files/CatBoost Regressor_Drift_Report.html", 'r', encoding='utf-8')
        source_code = html_file.read()
        components.html(source_code, height=3000, scrolling=True)

    # ref data
    df = get_row_data()
    df_cleaned = clean_data(df)

    # encoding (Options)
    df = df_cleaned.copy()
    df['Options'] = df['Options'].apply(literal_eval)
    df_opts = pd.DataFrame(mlb.transform(df['Options']), columns=mlb.classes_, index=df.index)
    df.drop("Options", axis=1, inplace=True)
    df = pd.concat([df, df_opts], axis=1)

    if uploaded_file is not None:
        with drift_report.container():
            new_df_cleaned = pd.read_csv(uploaded_file, encoding='latin1')
            new_df_cleaned = clean_data(new_df_cleaned)
            # encoding (Options)
            new_df_cleaned['Options'] = new_df_cleaned['Options'].apply(literal_eval)
            df_opts = pd.DataFrame(mlb.transform(new_df_cleaned['Options']), columns=mlb.classes_,
                                   index=new_df_cleaned.index)
            new_df_cleaned.drop("Options", axis=1, inplace=True)
            new_df_cleaned = pd.concat([new_df_cleaned, df_opts], axis=1)

            with st.spinner("Attendez un peu, cela peut prendre un certain temps..."):
                s = setup(data=df,
                          target='Prix',
                          test_data=new_df_cleaned,
                          max_encoding_ohe=100,
                          fold=5,
                          )

                predict_model(model, drift_report=True)

                html_file = open("CatBoost Regressor_Drift_Report.html", 'r', encoding='utf-8')
                source_code = html_file.read()
                components.html(source_code, height=3000, scrolling=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Drift des données",
                       page_icon="📈",
                       layout="wide")

    CSS = """ 
            <style> 
            button[title="View fullscreen"]{
            visibility: hidden;}
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

    st.markdown(CSS, unsafe_allow_html=True)

    expl()
