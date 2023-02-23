import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import bokeh_catplot
from bokeh.models import ColorBar, ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
import seaborn as sns
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


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


def eda():
    # for numerical data
    @st.cache(allow_output_mutation=True)
    def get_dist_plot(data, col):
        sigma, mu = data[col].std(), data[col].mean()
        hist, edges = np.histogram(data[col].dropna().tolist(), density=True, bins=100)

        np.isfinite(data[col])

        x = np.linspace(data[col].min(), data[col].max(), len(data[col].tolist()))
        pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        p = figure(width=550, height=300)
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#0051ff", line_color="white",
               alpha=0.5)
        p.line(x, pdf, line_color="#011d8a", line_width=4, alpha=0.7)
        p.y_range.start = 0
        p.xaxis.axis_label = col
        p.yaxis.axis_label = 'Density'
        p.grid.grid_line_color = "white"
        p.background_fill_color = "#ebf1f9"

        return p

    @st.cache(allow_output_mutation=True)
    def get_box_plot_num(data, col):
        p = bokeh_catplot.box(data=data, val=col, plot_width=550, plot_height=300)

        p.grid.grid_line_color = "white"
        p.background_fill_color = "#ebf1f9"

        # configure so that no drag tools are active

        return p

    # for categorical data
    @st.cache(allow_output_mutation=True)
    def get_vbar_plot(data, col, width, height):
        x = data[col].value_counts().index.tolist()
        y = data[col].value_counts().values.tolist()
        p = figure(x_range=x,
                   tooltips=[("Cars", "@x"), ("Count", "@top")],
                   plot_width=width,
                   plot_height=height)

        p.vbar(x=x, top=y, width=0.5)

        p.y_range.start = 0
        p.yaxis.axis_label = "Count"
        p.xaxis.axis_label = col
        p.grid.grid_line_color = "white"
        p.background_fill_color = "#ebf1f9"

        if col in hbar_cols:
            p.xaxis.major_label_orientation = np.pi / 6

        return p

    @st.cache(allow_output_mutation=True)
    def get_hbar_plot(data, col, width, height):
        x = data[col].value_counts().index.tolist()
        y = data[col].value_counts().values.tolist()
        x.reverse()
        y.reverse()

        p = figure(y_range=x,
                   tooltips=[("Cars", "@y"), ("Count", "@right")],
                   plot_width=width,
                   plot_height=height)

        p.hbar(y=x, right=y, height=0.5)

        p.xaxis.axis_label = "Count"
        p.grid.grid_line_color = "white"
        p.background_fill_color = "#ebf1f9"

        return p

    @st.cache(allow_output_mutation=True)
    def get_box_plot(data, col, val, width, height):
        data = data[data[val].notna()]

        p = bokeh_catplot.box(data=data,
                              cats=col,
                              val=val,
                              plot_width=width,
                              plot_height=height)

        p.grid.grid_line_color = "white"
        p.background_fill_color = "#ebf1f9"

        return p

    # scatter
    @st.cache(allow_output_mutation=True)
    def get_scatter_plot(data, x, y, hue):
        source = ColumnDataSource(data)
        p = figure()

        if hue not in "None":
            pal = sns.color_palette('RdBu', len(data[hue].unique()))
            mapper = linear_cmap(field_name=hue, palette=pal.as_hex(), low=data[hue].min(), high=data[hue].max())
            color_bar = ColorBar(color_mapper=mapper['transform'], width=30, location=(0, 0), title=hue)

            p.circle(x=x, y=y, line_color=mapper, color=mapper, fill_color=mapper, fill_alpha=1, size=3.5,
                     source=source)
            p.add_layout(color_bar, 'right')

        else:
            p.circle(x=x, y=y, fill_alpha=1, size=3.5,
                     source=source)

        p.xaxis.axis_label = x
        p.yaxis.axis_label = y
        p.grid.grid_line_color = "white"
        p.background_fill_color = "#ebf1f9"

        return p

    # pandas prof
    @st.cache(allow_output_mutation=True)
    def gen_profile_report(data, *report_args, **report_kwargs):
        return data.profile_report(*report_args, **report_kwargs)

    # main()
    # load data & gen pandas rapp
    df = get_row_data()
    with st.spinner('Attendez un peu...'):
        df_cleaned = clean_data(df)

    # cat & num cols
    categorical_cols = list(df_cleaned.select_dtypes(include=np.object).columns)
    categorical_cols.remove("Options")
    categorical_cols.remove("Model")
    numerical_cols = list(df_cleaned.select_dtypes(include=np.number).columns)

    # title
    st.title("📊 VISUALISATION DES DONNÉES")

    # Data distribution (nums & cats)
    tab1, tab2 = st.tabs(["La distribution des données numériques",
                          "La distribution des données catégorielles"])

    # numeric (Dist & Box)
    with tab1:
        col_num = st.columns(6)

        with col_num[0]:
            sb_num = st.selectbox("Sélectionner une variable :", sorted(numerical_cols), key="sb_num_dist")

        col1, _, col2 = st.columns([5, .1, 5])

        with col1:
            p_num_dist = get_dist_plot(df_cleaned, sb_num)
            st.bokeh_chart(p_num_dist, use_container_width=True)

        with col2:
            p_num_box = get_box_plot_num(df_cleaned, sb_num)
            st.bokeh_chart(p_num_box, use_container_width=True)

    # categorical (Bar & Box)
    with tab2:
        col_cat = st.columns(6)

        with col_cat[0]:
            sb_cat = st.selectbox("Sélectionner une variable :", sorted(categorical_cols), key="sb_cat_bar")

        with col_cat[1]:
            sb_cat_num_box = st.selectbox("Sélectionner une variable :", sorted(numerical_cols), key="sb_cat_num_box",
                                          index=2)

        st.markdown("")

        col1, _, col2 = st.columns([5, .1, 5])

        hbar_cols = ["Couleur", "Carrosserie"]

        with col1:
            if sb_cat in hbar_cols:
                p_cat_hbar = get_hbar_plot(df_cleaned, sb_cat, 550, 700)
                st.bokeh_chart(p_cat_hbar, use_container_width=True)
            elif sb_cat in 'Marque':
                p_cat_hbar = get_hbar_plot(df_cleaned, sb_cat, 550, 900)
                st.bokeh_chart(p_cat_hbar, use_container_width=True)
            else:
                p_cat_vbar = get_vbar_plot(df_cleaned, sb_cat, 550, 300)
                st.bokeh_chart(p_cat_vbar, use_container_width=True)

        with col2:
            if sb_cat in hbar_cols:
                p_cat_box = get_box_plot(df_cleaned, sb_cat, sb_cat_num_box, 550, 700)
                st.bokeh_chart(p_cat_box, use_container_width=True)
            elif sb_cat in 'Marque':
                p_cat_box = get_box_plot(df_cleaned, sb_cat, sb_cat_num_box, 550, 900)
                st.bokeh_chart(p_cat_box, use_container_width=True)
            else:
                p_cat_box = get_box_plot(df_cleaned, sb_cat, sb_cat_num_box, 550, 300)
                st.bokeh_chart(p_cat_box, use_container_width=True)

    # other graphs
    with st.form("parametres"):
        col_param, col_plot = st.columns([4, 20])

        with col_param:
            sb_x = st.selectbox("Axis-X :", sorted(numerical_cols), key="sb_x", index=1)
        with col_param:
            sb_y = st.selectbox("Axis-Y :", sorted(numerical_cols), key="sb_y", index=2)
        with col_param:
            sb_hue = st.selectbox("Variable de coloration (hue) :", sorted(numerical_cols) + ["None"], key="sb_hue",
                                  index=0)
        with col_param:
            button = st.form_submit_button("Plot")

        with col_plot:
            scatter_plot = st.empty()

        p_scatter = get_scatter_plot(df_cleaned, sb_x, sb_y, sb_hue)
        scatter_plot.bokeh_chart(p_scatter, use_container_width=True)

        if button:
            p_scatter = get_scatter_plot(df_cleaned, sb_x, sb_y, sb_hue)
            scatter_plot.bokeh_chart(p_scatter, use_container_width=True)

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

    prof_df = df_cleaned.reset_index(drop=True)
    profile = gen_profile_report(prof_df,
                                 missing_diagrams={
                                     "bar": False,
                                     "matrix": False,
                                     "heatmap": False,
                                     "dendrogram": False,
                                 },
                                 interactions={
                                     "continuous": False
                                 },
                                 samples={
                                     "head": False,
                                     "tail": False,
                                     "random": False
                                 }
                                 )

    with st.expander("RAPPORT : Pour des informations plus détaillées sur les données"):
        st_profile_report(profile)


if __name__ == "__main__":
    st.set_page_config(page_title="Visualisation des données",
                       page_icon="📊",
                       layout="wide")

    CSS = """ 
            <style> 
            button[title="View fullscreen"]{
            visibility: show;}
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

    st.markdown(CSS, unsafe_allow_html=True)

    eda()
