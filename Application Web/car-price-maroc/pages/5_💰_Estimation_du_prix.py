import streamlit as st
from PIL import Image
from pycaret.regression import *
import pandas as pd
import pickle
import joblib
from streamlit_shap import st_shap
import shap
from ast import literal_eval

# encoding
mlb = pickle.load(open('files/mlb.pkl', 'rb'))

# model
model = joblib.load('files/catboost_model.pkl')

# dictionary for storing data
data = {}


@st.cache(show_spinner=False)
def get_row_data():
    data = pd.read_csv("data/cars_data.csv", encoding='latin1')
    return data


@st.cache(show_spinner=False, allow_output_mutation=True)
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


df = get_row_data()
df_cleaned = clean_data(df)

# Marque List
Marque_List = df_cleaned["Marque"].unique().tolist()
# les Model Correspandant a chaque Marque (Dictionary)
Models_Dict = df_cleaned.groupby(["Marque"])["Model"].unique().apply(list).to_dict()
# Couleur List
Couleur_List = df_cleaned["Couleur"].unique().tolist()
# Carrosserie List
Carrosserie_List = df_cleaned["Carrosserie"].unique().tolist()
# List des options List
Options_List = mlb.classes_.tolist()


def predict_price():
    # dataframe
    df = pd.DataFrame(data, index=[0])

    # encoding (Options)
    df['Options'] = df['Options'].apply(literal_eval)
    df_opts = pd.DataFrame(mlb.transform(df['Options']), columns=mlb.classes_, index=df.index)
    df.drop("Options", axis=1, inplace=True)
    df = pd.concat([df, df_opts], axis=1)

    # make prediction
    unseen_predictions = predict_model(model, data=df)
    predicted_price = int(unseen_predictions["prediction_label"])
    predicted_price = predicted_price - (predicted_price % 1000)

    # show the result
    st.info("##### Le prix estimé de la voiture est : **{:,.2f} MAD**".format(predicted_price))

    # interpret the result
    tr_df = model.transform(df)
    explainer = shap.TreeExplainer(model[-1])
    shap_values = explainer.shap_values(tr_df)

    st_shap(shap.force_plot(explainer.expected_value, shap_values, tr_df))


def predict_price_form():
    cols = st.columns(2)
    Marque_List.sort()
    Marque_List.remove("AUTRE")
    Marque_List.append("AUTRE")
    data["Marque"] = cols[0].selectbox('Marque', Marque_List)

    ModelList = sorted(Models_Dict[data["Marque"]])
    if "Autre" not in ModelList:
        ModelList.append("Autre")
    else:
        ModelList.remove("Autre")
        ModelList.append("Autre")
    data["Model"] = cols[1].selectbox('Model', ModelList)

    cols = st.columns(3)
    data["Kilométrage"] = cols[0].number_input('Kilométrage', step=100, min_value=0)
    data["Année"] = cols[1].selectbox('Année', [*range(2022, 1939, -1)], key=3)
    data["Puissance fiscale"] = cols[2].number_input('Puissance fiscale', step=1, min_value=2, max_value=50)

    cols = st.columns(4)
    data["Boite de vitesses"] = cols[0].selectbox('Boite de vitesses', ('Automatique', 'Manuelle'))
    data["Carburant"] = cols[1].selectbox('Carburant', ('Diesel', 'Essence'))
    Couleur_List.sort()
    Couleur_List.remove("Autre")
    Couleur_List.append("Autre")
    data["Couleur"] = cols[2].selectbox('Couleur', Couleur_List)
    data["Carrosserie"] = cols[3].selectbox('Carrosserie', sorted(Carrosserie_List))

    cols = st.columns(2)
    data["Première main"] = "Oui" if cols[0].checkbox("Première main ?") else "Non"
    data["Véhicule dédouané"] = "Oui" if cols[0].checkbox("Véhicule dédouané ?") else "Non"
    data["Véhicule en garantie"] = "Oui" if cols[1].checkbox("Véhicule en garantie ?") else "Non"
    data["Importé neuf"] = "Oui" if cols[1].checkbox("Importé neuf ?") else "Non"

    data["Options"] = st.multiselect('Les Options :', Options_List)

    if st.button('Prédire'):
        data["Options"] = str(data["Options"])

        predict_price()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Estimation du prix",
        page_icon="💰",
        layout="centered"
    )

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

    img_car_price = Image.open("imgs/car_price.png")

    col, _1, _2, _3 = st.columns([16, 1, 4, 1])
    with col:
        st.title("PRÉDICTION DU PRIX DES VOITURES D'OCCASION:")
    with _2:
        st.image(img_car_price, width=160)

    st.markdown('**Objectif** : Étant donné les détails sur la voiture, le modèle va essayer de prédire le prix.')

    predict_price_form()
