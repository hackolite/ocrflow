from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def get_node_id(latitude, longitude):
    geolocator = Nominatim(user_agent="my_geocoder")
    try:
        location = geolocator.reverse((latitude, longitude), exactly_one=True)
        print(location)
        if location.raw.get('osm_id'):
            print(location.raw)
            return location.raw['osm_id']
        else:
            print("No OSM ID found for the provided coordinates.")
            return None
    except GeocoderTimedOut:
        print("Geocoding service timed out. Please try again later.")
        return None

# Coordonnées géographiques (latitude et longitude)
latitude = 48.9286
longitude = 2.510
#48.92863829366043, 2.5101166591406323

#48.87684/2.34812
#48.86919/2.33774




# Récupérer le Node ID
node_id = get_node_id(latitude, longitude)
print("Node ID:", node_id)
