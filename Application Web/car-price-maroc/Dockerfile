FROM python:3.9

EXPOSE 9050 
EXPOSE 8501

WORKDIR /car-price-maroc

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

RUN python generate_dashboard.py

CMD python run_dashboard.py & streamlit run 0_🏠_Accueil.py