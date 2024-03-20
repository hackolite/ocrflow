from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# importing ObjectId from bson library
from bson.objectid import ObjectId
from pymongo import MongoClient
from datetime import datetime
import pytz
from datetime import datetime
import random




class MongoConnector:

    def __init__(self, user="xretail", password="kodjemana", host="cluster0.nn3l2bm.mongodb.net"):
        #self.connection = 
        uri = "mongodb+srv://{}:{}@{}/".format(user, password, host)
        # Create a new client and connect to the server
        client = MongoClient(uri, server_api=ServerApi('1'))
        self.connection = client 


    def collection(self):
        db = self.connection['XRETAIL']
        collection = db['PRICE']
        self.collection = collection


    def store_mongodb(self, ean=None, description=None, price=None, gtin=False, store_id=None, session=None, user=None, url_image=None, annotation=None, queue=None):
        utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
        document= {"ean" : ean, "description": "DEMO", "price": price, "time": utc_now, 
                    "valid":gtin, "user":user, "session":session, "project_id":session, "store_id":store_id, "url_image":url_image, "annotation":annotation, "queue":queue}
        # Insert the data into the collection
        result = self.collection.insert_one(document)
        # Print the inserted document's ID
        print("Inserted document ID:", result.inserted_id)


    def delete_all(self):
        # Delete all documents from the collection
        result = self.collection.delete_many({})
        print("Deleted", result.deleted_count, "documents")


    def declare_session(self, queue=None):
        db = self.connection['XRETAIL']
        session_collection = db['SESSION']
        document= {"session" : queue, "email":False}
        # Define the filter to check if the document already exists
        filter = {'session': queue}
        # Define the update operation
        update = {'$set': document}
        # Perform the update operation with upsert=True
        result = session_collection.update_one(filter, update, upsert=True)
        # Check if the document was inserted or updated
        if result.upserted_id:
            print("Document inserted:", result.upserted_id)
        else:
            print("Document already exists")



    def close_session(self, queue=None):
        db = self.connection['XRETAIL']
        session_collection = db['SESSION']
        document= {"session" : queue, "email":True}
        # Define the filter to check if the document already exists
        # Define the update operation
        filter = {'session': queue}
        update = {'$set': document}
        # Perform the update operation with upsert=True
        result = session_collection.update_one(filter, update)
        # Check if the document was inserted or updated


    def is_email_sent(self, queue=None):
        db = self.connection['XRETAIL']
        session_collection = db['SESSION']
        document= {"session" : queue}
        results = session_collection.find(document)
        for result in results:
            if result["email"] == True:
                return True
        return False

    def get_queue_session(self, queue=None):
        db = self.connection['XRETAIL']
        price_collection = db['PRICE']
        document = {"queue" : queue}
        results = price_collection.find(document)
        return results
