import requests

# Définir les coordonnées (latitude et longitude)
latitude = 48.9289310  # Exemple de latitude (Tour Eiffel, Paris)
longitude = 2.5101580  # Exemple de longitude



# Construire l'URL de la requête Nominatim pour une recherche par bâtiment
url = "https://nominatim.openstreetmap.org/search"
params = {
    'q': 'supermarket',
    'format': 'json',
    'lat': latitude,
    'lon': longitude,
    'addressdetails': 1,
    'limit': 10,  # Limiter à 10 résultats
    'polygon_geojson': 1  # Inclure les données géométriques du bâtiment
}

# Effectuer la requête
response = requests.get(url, params=params)
data = response.json()

# Afficher les résultats des bâtiments trouvés
print("Bâtiments trouvés à proximité:")
for result in data:
    print(f"Nom: {result.get('display_name', 'N/A')}")
    print(f"Type: {result.get('type', 'N/A')}")
    print(f"ID OSM: {result.get('osm_id', 'N/A')}")
    print(f"Coordonnées: {result.get('lat')}, {result.get('lon')}")
    if 'geojson' in result:
        print(f"Géométrie: {result['geojson']}")
    print()
