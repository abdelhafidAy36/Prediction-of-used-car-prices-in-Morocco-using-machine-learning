import streamlit as st
import streamlit.components.v1 as components


def expl():
    st.title("🔍 EXPLICABILITÉ DU MODÈLE")

    st.markdown("")
    st.markdown("")

    components.iframe("http://localhost:9050/", height=3000, scrolling=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Explicabilité du modèle",
                       page_icon="🔍",
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
