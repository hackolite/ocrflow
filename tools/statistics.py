from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import requests
# importing ObjectId from bson library
from bson.objectid import ObjectId

from pymongo import MongoClient
from datetime import datetime
import pytz
from datetime import datetime
import random
from string import digits
from gtin import has_valid_check_digit, calculate_check_digit
import pandas as pd



POIDS = []
PRIX = []
PRIXKG = []
MARQUE = []
EAN = []




def get_product_info(barcode):
    # Base URL of the Open Food Facts API
    base_url = "https://world.openfoodfacts.org/api/v0/product/"
    # Construct the complete URL with the barcode
    url = base_url + str(barcode) + ".json"
    # Send a GET request to the API
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    print(response.status_code)
    if response.status_code == 200:
        # Parse the JSON response
        product_data = response.json()
        try:
            if 'quantity' in product_data['product']:
                brand = product_data['product']["brands"]
                if ('pate' in product_data["product"]['_keywords']) or ("pasta" in product_data["product"]['_keywords']):
                    weight = product_data['product']['product_quantity']
                    result = {"poids": round(float(weight.replace("g","")),2), "marque": brand}
                    print("RESULT :", result)
                    return result 
                else:
                    return {"poids":None, "marque": None}
            else:
                return {"poids":None, "marque": None}

        except:
            return {"poids":None, "marque": None}



uri = "mongodb+srv://xretail:kodjemana@cluster0.nn3l2bm.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
# Sélection de la base de données (elle sera créée si elle n'existe pas)
db = client['XRETAIL']
# Select or create a collection
collection = db['PRICE']

count = 0
for document in collection.find():
    count += 1
    if len(document["ean"]) == 13:
        if has_valid_check_digit(document["ean"]):
            resultat = get_product_info(document["ean"])
            print(resultat)
            if (resultat["poids"] == None) : 
                pass    
            
            else:
                    try:
                        EAN.append(document["ean"])

                        poids = float(resultat["poids"])
                        if poids == 0:
                            poids = 10000000000
                        POIDS.append(poids)
                        

                        try:
                            price = float(document["price"].replace("i","1"))
                        except:
                            price = 0    

                        if price == 0:
                            PRIX.append(1000000000)
                        else:
                            PRIX.append(price)
                        
                        PRIXKG.append(round(price/poids*1000,2))
                        try:
                            MARQUE.append(resultat["marque"])
                        except:
                            MARQUE.append("")
                        count += 1
                    except Exception as e:
                        print(e)




# initialize data of lists.
data = {'EAN': EAN, 'POIDS': POIDS, 'PRIX': PRIX, 'PRIX-KG': PRIXKG, 'MARQUE': MARQUE}

print("EAN :", len(EAN))
print("POIDS :", len(POIDS))
print("PRIX :", len(PRIX))
print("PRIX-KG :", len(PRIXKG))
print("MARQUE :", len(MARQUE))


# Creates pandas DataFrame.
market = pd.DataFrame(data)
market.to_csv("market.csv")



