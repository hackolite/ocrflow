from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# importing ObjectId from bson library
from bson.objectid import ObjectId

from pymongo import MongoClient
from datetime import datetime
import pytz
from datetime import datetime
import random
from pymongo import MongoClient




from pymongo import MongoClient
from pymongo.errors import OperationFailure

# Adresse IP du serveur MongoDB
adresse_ip = '139.99.9.198'

# Ancien et nouveau mot de passe root
ancien_mot_de_passe = 'example'
nouveau_mot_de_passe = 'chcocoBoule972973974'

# Connexion au serveur MongoDB
client = MongoClient(f"mongodb://root:{nouveau_mot_de_passe}@{adresse_ip}:27017/")

db = client.admin


# Define the new user's information
new_username = 'xretail'
new_password = 'liaterx'

# Create the new user
#db.command('createUser', new_username, pwd=new_password, roles=['readWrite'])

print("User '{}' created successfully.".format(new_username))


from pymongo import MongoClient

# Connect to MongoDB server
client = MongoClient('mongodb://localhost:27017/')

# Access or create a database
db = client['off']  # Replace 'mydatabase' with the name of your database




# Close the connection
client.close()