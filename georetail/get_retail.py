import requests
from get_address import get_address


# Définition de la requête Overpass pour les supermarchés et hypermarchés
overpass_url = "http://overpass-api.de/api/interpreter"
overpass_query = """
    [out:json];
    area["name"="Seine-Saint-Denis"];
    (
      node["shop"="supermarket"](area);
      node["shop"="hypermarket"](area);
    );
    out;
"""

# Envoi de la requête à l'API Overpass
response = requests.get(overpass_url, params={'data': overpass_query})

# Vérification du code de statut de la réponse
if response.status_code == 200:
    # Récupération des données au format JSON
    data = response.json()
    # Affichage des résultats
    for i in data['elements']:
        try:
            add = [i["tags"]["name"]] + get_address(i["lat"], i["lon"]).split(",")[:-6]
        except:
            add = [i["tags"]["shop"]] + get_address(i["lat"], i["lon"]).split(",")[:-6]
        print(add[0], ";", " ".join(add[0:-2]), ";", add[-1])


else:
    print("Erreur lors de la requête : ", response.status_code)

