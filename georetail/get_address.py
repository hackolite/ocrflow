import requests



def get_address(latitude, longitude):
    # Coordonnées de latitude et longitude
    # URL de l'API de géocodage inversé de Nominatim
    nominatim_url = "https://nominatim.openstreetmap.org/reverse"

    # Paramètres de la requête
    params = {
    'format': 'json',
    'lat': latitude,
    'lon': longitude,
    }

    # Envoi de la requête à l'API de géocodage inversé de Nominatim
    response = requests.get(nominatim_url, params=params)

    # Vérification du code de statut de la réponse
    if response.status_code == 200:
        # Récupération des données au format JSON
        data = response.json()
        address_parts = data.get('address', {})
        address_parts.pop('country', None)
        address_parts.pop('state', None)



        #address = ", ".join(filter(None, address_parts.values()))
        # Récupération de l'adresse complète
        address = data.get('display_name', 'Adresse non trouvée')
        # Affichage de l'adresse
    else:
        address = ""
        print("Erreur lors de la requête : ", response.status_code)

    return address




from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="specify_your_app_name_here")
location = geolocator.reverse("52.509669, 13.376294")
print(location.address)
print((location.latitude, location.longitude))
print(location.raw)
{'place_id': '654513', 'osm_type': 'node', ...}