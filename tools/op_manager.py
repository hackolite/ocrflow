#helper pour  openprices
import requests
import xlsxwriter
from datetime import datetime



service_url = "prices.openfoodfacts.net"

def connect():
	pass


def get_data():
	resp = connection("laureote", "kodjemana972*")
	token = resp.json()['access_token']
	headers = {'Authorization': 'Bearer {}'.format(token)}
    
	for price_id in values:
			price_id = str(price_id)
			url = "https://{}/api/v1/prices/{}".format(service_url, price_id)


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




