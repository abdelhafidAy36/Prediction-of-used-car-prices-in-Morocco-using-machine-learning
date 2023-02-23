import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid


@st.cache(show_spinner=False)
def get_row_data():
    data = pd.read_csv("data/cars_data.csv", encoding='latin1')
    return data


@st.cache(show_spinner=False)
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


def description_des_donnees():
    def ag_grid_interactive_table(data):
        gb = GridOptionsBuilder.from_dataframe(data)

        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="avg", editable=True)

        grid_options = gb.build()

        AgGrid(data,
               gridOptions=grid_options,
               theme='streamlit',
               enable_enterprise_modules=True,
               width='100%')

    st.title("📝 DESCRIPTION DES DONNÉES")

    col1, _1, col2 = st.columns([3, 10, 3])

    with col1:
        dt = st.selectbox('les données :', ('les données brutes', 'les données nettoyées'),
                          label_visibility="collapsed")

    #  dataframe
    df = get_row_data()
    if dt == "les données brutes":
        ag_grid_interactive_table(df)
    else:
        with st.spinner('Attendez un peu...'):
            df_cleaned = clean_data(df)
        ag_grid_interactive_table(df_cleaned)

    # expander css
    st.markdown(
        """
    <style>
    .streamlit-expanderHeader {
        font-size: x-large;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Les Colonnes
    with st.expander("Voir la description des données"):
        st.write(""" 
        - **Name :** Le nom de la voiture sous la form (marque + model).
        - **Kilométrage :** Le kilométrage parcouru.
        - **Année :** L’année de fabrication.
        - **Boite de vitesses :** Le type de voiture (Automatique ou Manuelle).
        - **Carburant :** Le type de carburant, qu’il s’agisse d’essence, diesel ou autre.
        - **Date :** Date d’annonce sur le site.
        - **Puissance fiscale :** Unité administrative calculée, en partie, à partir de la puissance réelle du moteur.
        - **Nombre de portes :** Nombre de portes de la voiture.
        - **Première main :** Acheter directement de la source originale.
        - **Voiture personnalisée (tuning) :** La voiture a été modifié pour l'optimiser en fonction d'un ensemble d'exigences.
        - **Véhicule dédouané :** Véhicule dédouané ou non. 
        - **Véhicule en garantie :** Véhicule en garantie ou non.
        - **Importé neuf :**  Importé de l'extérieur du pays à l'état neuf.
        - **Couleur :** Couleur de la voiture.
        - **Carrosserie :** Il indique le type de voiture, par exemple une voiture "Suv et 4x4"ou une autre catégorie.
        - **Options :** List des options du voiture.
        - **Prix :** Le prix de voiture.
        """)

    # code
    with st.expander("Voir le web scraper utilisé pour extraire les données"):
        code = r"""
# -*- coding: UTF-8 -*-
from bs4 import BeautifulSoup
import requests
import multiprocessing
import csv
    
fieldnames = ["Nom", 'Kilométrage', 'Année', 'Boite de vitesses', 'Carburant', 'Date', 'Puissance fiscale',
              'Nombre de portes', 'Première main', 'Voiture personnalisée (tuning)', 'Véhicule dédouané',
              'Véhicule en garantie', 'Importé neuf', 'Couleur', 'Carrosserie', 'Options', 'Prix']
    
csv_file = "CarsRowData.csv"

all_urls = []
    
base_url = "https://www.moteur.ma/fr/voiture/achat-voiture-occasion/"
    
    
def generate_urls():
    for i in range(0, 30616, 15):
        all_urls.append(base_url + str(i))
    

def scraping_cars(url):
    # open file
    f = open(csv_file, 'a', newline='', encoding="latin1")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'lxml')
    
    All_Cars = soup.find_all('div', class_="row-item row-item-checkout link")
    
    for car in All_Cars:
        row = {}
    
        try:
            html_text_in = requests.get(car.a['href']).text
            soup2 = BeautifulSoup(html_text_in, 'lxml')
        except:
            continue
    
        # Nom (Car Name)
        try:
            nom = soup2.find("div", class_="col-md-12 col-sm-12 col-xs-12 text-center ads-detail")
            nom = nom.h1.span.text.replace('\n', '').replace('\t', '').strip()
            row["Nom"] = nom
        except:
            continue
    
        # Prix (Price)
        try:
            prix = soup2.find("div", class_="color_primary text_bold price-block")
            prix = prix.text.replace('\n', '').replace('\t', '').strip()
            row["Prix"] = prix
        except:
            continue
    
        # Autres Colonnes (Other Columns)
        try:
            boxs = soup2.find_all('div', class_="detail_line")
            for box in boxs:
                col = box.find('span', class_="col-md-6 col-xs-6").text.replace('\n', '').replace('\t', '').strip()
                val = box.find('span', class_="text_bold").text.replace('\n', '').replace('\t', '').replace('\u200b', '').strip()
    
                row[col] = val
        except:
            continue
    
        # Options de voiture (Car Options)
        try:
            opts = soup2.find_all('div', class_="col-md-6 option_ad")
    
            ListOfOptions = []
            for opt in opts:
                option = opt.text.replace("✔", "").replace('\n', '').replace('\t', '').strip()
                ListOfOptions.append(option)
    
            row["Options"] = ListOfOptions
        except:
            pass
    
        print(row)
        writer.writerow(row)
    
    f.close()
    
    
if __name__ == '__main__':
    # open file and write columns name in the first row
    f = open(csv_file, 'w', newline='')
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    f.close()
    
    # generate  urls
    generate_urls()
    
    # scraping using multi-threading
    pool = multiprocessing.Pool(8)
    pool.map(scraping_cars, all_urls)
    pool.close()
    """

        _, col, _ = st.columns([.5, 6, .5])
        with col:
            st.code(code, language="python")


if __name__ == "__main__":
    st.set_page_config(page_title="Description des données",
                       page_icon="📝",
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

    description_des_donnees()
