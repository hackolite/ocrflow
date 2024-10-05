#helper pour  openprices
import requests
import xlsxwriter
from datetime import datetime
import time


service_url = "prices.openfoodfacts.org"


user = "laureote"
password = "kodjemana972*"

import requests

	

# Paramètres de la requête
params = {
    "owner": "laureote",
    "json": True , # Format de réponse JSON
}


data = {
    'date': ''
}

count = 0

def connection(user, password):

	headers = {'accept': 'application/json','Content-Type': 'application/x-www-form-urlencoded'}

	data = {
		'set_cookie':'true',
	    'grant_type': '',
	    'username': user,
	    'password': password,
	    'scope': '',
	    'client_id': '',
	    'client_secret': ''
	}

	url = "https://{}/api/v1/auth".format(service_url)
	token = requests.post(url, data=data, headers=headers)
	print("token", token)
	return token


def get_data():
	count = 1
	resp = connection(user, password)
	token = resp.json()['access_token']
	headers = {'Authorization': 'Bearer {}'.format(token)}
	url = "https://{}/api/v1/prices".format(service_url)
	while True:
		params["page"] = count
		response = requests.get(url, params=params)
		items = response.json()["items"]
		
		if len(items) > 0:
			for i in items:
				if "2023" in i["date"]:
					print(i["date"], i["created"])
					date = i["date"].replace("2023", "2024")
					data["date"] = date
					print(data)
					url_mod = url + "/"+ str(i["id"])
					resp = requests.patch(url_mod, headers=headers, json=data)
					print(resp.reason)
			count +=1 
		else:
			return 
get_data()


def update_data():
	pass