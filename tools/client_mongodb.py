from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# importing ObjectId from bson library
from bson.objectid import ObjectId

from pymongo import MongoClient
from datetime import datetime
import pytz
from datetime import datetime
import random



# Get the current date and time
current_datetime = datetime.now()
# Format the current date and time as ISO 8601 string
iso_date_string = current_datetime.isoformat()

uri = "mongodb+srv://xretail:@cluster0.nn3l2bm.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection




def test_connection():
    
    try:
            client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
            print(e)
    
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    

    # Sélection de la base de données (elle sera créée si elle n'existe pas)
    db = client['XRETAIL']
    # Création de la collection
    collection_name = 'PRICE'
    # Vérification de la création de la collection
    collection_names = db.list_collection_names()
    if collection_name in collection_names:
        print(f"La collection {collection_name} a été créée avec succès.")
    else:
        print(f"La collection {collection_name} n'a pas pu être créée.")

    FLOAT = round(random.uniform(10.5, 75.5), 2)
    utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)

    print(utc_now)
    document= {
        "product_id": ObjectId("65a3ed6673703b56a5ef0610"),
        "store_id": ObjectId("65a3e9cf73703b56a5ef060f"),
        "description": "Description du produit A",
        "price": FLOAT,
        "time": utc_now,
    }

    db = client['XRETAIL']
    # Select or create a collection
    collection = db['PRICE']

    # Insert the data into the collection
    result = collection.insert_one(document)
    # Print the inserted document's ID
    print("Inserted document ID:", result.inserted_id)
    # Fermeture de la connexion
    client.close()
    #delete all entry monogdb 


def delete_all():
    db = client['XRETAIL']
    collection = db['PRICE']
    collection.delete_many({})

if __name__ == '__main__':
    delete_all()
