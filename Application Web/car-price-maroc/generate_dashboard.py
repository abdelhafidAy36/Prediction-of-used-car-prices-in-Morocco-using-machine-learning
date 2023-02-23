from pycaret.regression import *
import pandas as pd
import dash_bootstrap_components as dbc
from ast import literal_eval
import pickle
from explainerdashboard import RegressionExplainer, ExplainerDashboard

# encoding
mlb = pickle.load(open('files/mlb.pkl', 'rb'))

# model
model = load_model('files/tuned_catboost')


def ml_data():
    # load data
    df = pd.read_csv("data/cars_data.csv", encoding='latin1')

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

    # options
    df['Options'] = df['Options'].apply(literal_eval)
    df_opts = pd.DataFrame(mlb.transform(df['Options']), columns=mlb.classes_, index=df.index)
    df.drop("Options", axis=1, inplace=True)
    df = pd.concat([df, df_opts], axis=1)

    return df


ml_data = ml_data()

s = setup(data=ml_data,
          target='Prix',
          train_size=0.8,
          max_encoding_ohe=100,
          fold=5,
          session_id=49
          )

X_test, y_test = get_config("X_test"), get_config("y_test")

explainer = RegressionExplainer(model, X_test, y_test, units="MAD")
db = ExplainerDashboard(explainer,
                        title="",
                        whatif=False,
                        depth=20,
                        hide_contributiontable=True,
                        bootstrap=dbc.themes.FLATLY
                        )
db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)
