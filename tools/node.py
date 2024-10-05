import requests

# Define the Nominatim API endpoint and parameters
nominatim_url = "https://nominatim.openstreetmap.org/reverse"
params = {
    'format': 'jsonv2',
    'lat': 48.93,
    'lon': 2.50
}


# Send the request to the Nominatim API
response = requests.get(nominatim_url, params=params)

# Parse the response
data = response.json()

# Print the result
print(data)
