# -*- coding: UTF-8 -*-
from bs4 import BeautifulSoup
import requests
import multiprocessing
import csv

fieldnames = ["Nom", 'Kilométrage', 'Année', 'Boite de vitesses', 'Carburant', 'Date', 'Puissance fiscale',
              'Nombre de portes', 'Première main', 'Voiture personnalisée (tuning)', 'Véhicule dédouané',
              'Véhicule en garantie', 'Importé neuf', 'Couleur', 'Carrosserie', 'Options', 'Prix']

csv_file = "MoutuersCarsRowData_19_08.csv"

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

        # Nom
        try:
            nom = soup2.find("div", class_="col-md-12 col-sm-12 col-xs-12 text-center ads-detail")
            nom = nom.h1.span.text.replace('\n', '').replace('\t', '').strip()
            row["Nom"] = nom
        except:
            continue

        # Prix
        try:
            prix = soup2.find("div", class_="color_primary text_bold price-block")
            prix = prix.text.replace('\n', '').replace('\t', '').strip()
            row["Prix"] = prix
        except:
            continue

        # Box (Autres Colonnes)
        try:
            boxs = soup2.find_all('div', class_="detail_line")
            for box in boxs:
                col = box.find('span', class_="col-md-6 col-xs-6").text.replace('\n', '').replace('\t', '').strip()
                val = box.find('span', class_="text_bold").text.replace('\n', '').replace('\t', '').replace('\u200b', '').strip()

                row[col] = val
        except:
            continue

        # Options
        try:
            opts = soup2.find_all('div', class_="col-md-6 option_ad")

            ListOfOptions = []
            for opt in opts:
                option = opt.text.replace("✔", "").replace('\n', '').replace('\t', '').strip()
                ListOfOptions.append(option)

            if len(ListOfOptions) > 0:
                row["Options"] = ListOfOptions
            else:
                row["Options"] = None
        except:
            continue

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
