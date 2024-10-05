from geopy.geocoders import Nominatim
import overpy

# Initialiser le géocodeur Nominatim
geolocator = Nominatim(user_agent="GeoApp/1.0 (myemail@exemple.com)")

# L'adresse que tu veux géocoder
adresse = "2-20 All. de l'Est, 93190 Livry-Gargan"

# Récupérer les coordonnées (latitude, longitude) de l'adresse
location = geolocator.geocode(adresse)

if location:
    print(f"Adresse: {location.address}")
    print(f"Latitude: {location.latitude}, Longitude: {location.longitude}")
    
    # Utiliser l'API Overpass pour trouver l'OSM_ID du point correspondant
    api = overpy.Overpass()

    query = f"""
    node(around:10,{location.latitude},{location.longitude});
    out body;
    """

    # Effectuer la requête
    result = api.query(query)

    # Extraire les informations des nodes proches
    if result.nodes:
        node = result.nodes[0]  # Le premier nœud proche
        print(f"OSM_ID du nœud le plus proche: {node.id}")
    else:
        print("Aucun nœud trouvé près de cette adresse.")
else:
    print("Adresse non trouvée.")
