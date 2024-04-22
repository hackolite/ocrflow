import requests

# Step 1: Get Coordinates from Nominatim
# Step 1: Define the Coordinates
latitude = 48.8566  # Example latitude (Paris, France)
longitude = 2.3522  # Example longitude (Paris, France)

# Step 2: Perform Reverse Geocoding to Find Nearby Shops
nominatim_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}&zoom=18&addressdetails=1"
response = requests.get(nominatim_url)
data = response.json()

print(data)

latitude = data["lat"]
longitude = data["lon"]


# Step 2: Search for Nearby Shops
radius = 10  # Radius in meters (adjust as needed)
shop_query = "shop"  # You can adjust the query based on the type of shops you're interested in
nominatim_url = f"https://nominatim.openstreetmap.org/search?format=json&q={shop_query}&lat={latitude}&lon={longitude}&radius={radius}"
response = requests.get(nominatim_url)
shops_data = response.json()



# Step 3: Identify the Nearest Shop
print(shops_data)
