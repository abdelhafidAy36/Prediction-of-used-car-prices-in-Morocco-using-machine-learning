import streamlit as st
from PIL import Image


def run():
    _1, center, _2 = st.columns([2, 7, 2])

    with center:
        img_car_price = Image.open("imgs/home_page.PNG")
        st.image(img_car_price, use_column_width=True)

    _1, center, _2 = st.columns([2, 9, 2])

    with _1:
        st.markdown("")
    with center:
        st.write("")
        st.markdown(
            """
                <div style="text-align: justify; font-size:18px; ">
                L'achat ou la vente <strong>d'une voiture d'occasion</strong> est un processus difficile car il exige un 
                effort et des connaissances notables de la part de l'expert en la matière. Un nombre considérable d'attributs
                distincts sont examinés pour une prédiction fiable et précise. Pour cela, nous décidons d'utiliser des techniques
                <strong>d'apprentissage automatique</strong> pour construire un modèle de prédiction précis des prix 
                des voitures d'occasion au <strong>Maroc</strong>. Ensuite, Le modèle de prédiction final a été intégré
                 dans une application web, Lorsqu'il peut être utilisé par une société spécialisée dans la vente et 
                 l'achat de voitures.

                <H3>L'application se compose de quatre pages :</H3>
                <ul >
                <li style="text-align: justify; font-size:18px; "> <strong style="font-size:18px;">Description des données :</strong> Cette page contient les données utilisées pour construire notre modèle ML. Aussi une description des données et le "web scraper" écrit en langage python utilisé pour extraire les données. </li>
                <li style="text-align: justify; font-size:18px; "> <strong style="font-size:18px;">Visualisation des données :</strong> Cette page permet de représenter visuellement les données (statistiques descriptives, graphiques...). </li>
                <li style="text-align: justify; font-size:18px; "> <strong style="font-size:18px;">Explication du modèle :</strong> Cette page affiche différentes figures permettant d'analyser les performances du modèle.</li>
                <li style="text-align: justify; font-size:18px; "> <strong style="font-size:18px;">Drift des données :</strong> Cette page permet de détecter et d'explorer les changements dans les données d'entrée.</li>
                <li style="text-align: justify; font-size:18px; "> <strong style="font-size:18px;">Estimer une voiture :</strong> Dans cette page, nous pouvons faire une prédiction du prix d'une voiture d'occasion en fonction de ses caractéristiques et aussi interpréter la prédiction.</li>
                </ul>
                </div>
            """
            , unsafe_allow_html=True
        )


    with _2:
        st.markdown("")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Accueil",
        page_icon="🏠",
        layout="wide"
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

    run()
