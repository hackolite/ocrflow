import pandas as pd
import re

import requests

def get_product_info(barcode):
    # Base URL of the Open Food Facts API
    base_url = "https://world.openfoodfacts.org/api/v0/product/"

    # Construct the complete URL with the barcode
    url = base_url + str(barcode) + ".json"

    # Send a GET request to the API
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        product_data = response.json()
        # Check if weight information is available
        if 'quantity' in product_data['product']:
                #print(product_data['product'])
                brand = product_data['product']["brands"]
                #print(name)
                weight = product_data['product']['product_quantity']
                #print(f"Weight: {weight}")
                # Regular expression pattern to match floats
                #print(product_data['product']['brand'])
                #if "kg" in weight:
                #	weight = weight.replace("kg","")
                #	#weight = weight.replace("g","")
                #	#print(weight, "KG")

                #else:
                #	weight = weight.replace("g","")
                #	#print(weight, "G")
                #	weight = float(weight)/1000
                #print(weight)
                return {"poids": round(float(weight),2), "marque": brand}


# Example barcode (you can replace this with the actual barcode you want to search for)
barcode = "737628064502"

# Call the function to get product information
#get_product_info(barcode)



df = pd.read_excel("auchan.xlsx", header=0)
for i, value in df.iterrows():
	#print(value)
	barcode = value["gtin"]
	price = value["price"]
	resultat  = get_product_info(barcode)
	try:
		print(round(price/resultat["poids"]*1000,2), resultat["marque"])
	except Exception as e:
		pass
