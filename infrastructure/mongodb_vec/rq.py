import requests

# Faire une requête GET à une URL
response = requests.get('http://139.99.9.198:5001/data')

# Vérifier si la requête a réussi (code d'état HTTP 200)
if response.status_code == 200:
    # Afficher le contenu de la réponse
    print(response.text)
else:
    # Afficher un message d'erreur si la requête a échoué
    print(f'Erreur: {response.status_code}')
