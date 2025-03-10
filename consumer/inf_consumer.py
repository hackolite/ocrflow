import sys
sys.path.append("..")


import pika
import configuration
from connector.connector_mongodb import MongoConnector
from connector.connector_xls import XlsConnector
import json 
from gtin import has_valid_check_digit
import time
import random
import requests
from bson.objectid import ObjectId
from ocr import ocr
import smtplib
from email.mime.text import MIMEText
from send import mail
import traceback
from configuration import *



mongo = MongoConnector(user=mongodb_credentials["user"] , password=mongodb_credentials["password"], host=mongodb_credentials["host"])
mongo.collection()
data = ocr()


subject = "XRETAIL JOBS"
sender = email_credentials["user"]
recipient = sender
username = email_credentials["user"]
password = email_credentials["password"]


rbq_host         = rabbitmq_credentials["host"]
rbq_user         = rabbitmq_credentials["user"]
rbq_password     = rabbitmq_credentials["password"]





#parse a string a remove \
def filter_message(message):
    message = message.replace("\\","")
    return message


def get_random_string(string=10):
    string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=string))
    return string


def get_queue_message_count(queue_name, host, username, password):
    # RabbitMQ Management API URL
    url = 'http://{}:15672/api/queues/%2F/{}'.format(rbq_host, queue_name)
    response = requests.get(url, auth=(rbq_user, rbq_password))
    # Check if request was successful
    if response.status_code == 200:
        queue_info = response.json()
        message_count = queue_info['messages']
        return message_count 
    else:
        return None


def get_queues_list(host, username, password):
    # URL de l'API de gestion RabbitMQ
    url = 'http://{}:15672/api/queues'.format(host)
    # Informations d'identification (remplacez par vos propres informations d'identification si nécessaire)
    auth = (rbq_user, rbq_password)
    # Demander la liste des files RabbitMQ
    response = requests.get(url, auth=auth)
    return response.json()



# Callback function to handle incoming messages
def callback(ch, method, properties, body):
    global job 
    global email
    try:
        print("Received message:", body.decode())  # Assuming the message is in UTF-8 encoding
        resp = json.loads(body)
        email = resp["user_id"]
        shop_id = resp["shop_id"]
        session = resp["session"]
        url_image = resp["url_image"]
        result = data.process(resp["url_image"])
        
        if len(result["ean"]) == 13:
            is_valid = has_valid_check_digit(result["ean"])
        else:
            is_valid = False
        
        mongo.store_mongodb(ean=result["ean"], description="PRODUCTION", price=result["price"], user=email, gtin=is_valid, 
            store_id= shop_id, session=session, url_image=url_image, annotation=result["annotation"], queue=job)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        res = ch.queue_delete(job, if_empty=True)
        # Vérifiez si la file d'attente est vide et fermez-la si nécessaire
        # Add your message processing logic here
    except Exception as e:
        traceback.print_exc()


def on_queue_cancelled():
    #queue_name = method_frame.method.queue
    print(f"La file '{queue_name}' a été fermée par le serveur.")



def connect_to_queue(queue_name):
    # Établir la connexion
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    # Consommer les messages de la nouvelle queue
    channel.basic_consume(queue=queue_name, on_message_callback=callback)
    print(f"Connecté à la nouvelle queue: {queue_name}")
    channel.start_consuming()



def consume_with_retry(queue_name):
    while True:
        try:
            print("Attempting to connect and start consuming messages...")
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.basic_consume(queue=queue_name, on_message_callback=callback)
            print("Connected and started consuming messages.")
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError:
            print("Connection error. Retrying in 5 seconds...")
            time.sleep(5)




credentials = pika.PlainCredentials(rbq_user, rbq_password)
parameters = pika.ConnectionParameters(rbq_host, 5672, '/', credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()


while True:
    time.sleep(5)  # Attendre quelques secondes avant de vérifier à nouveau
    print("waiting for queue")
    # Vérifier à nouveau la liste des queues
    try:
        resp = get_queues_list(rbq_host, rbq_user,rbq_password)
    except:
        credentials = pika.PlainCredentials(rbq_user, rbq_password)
        parameters = pika.ConnectionParameters(rbq_host, 5672, '/', credentials)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        resp = []

    list_queue = []
    for i in resp:
        list_queue.append(i["name"])
    if list_queue:
        print("connect to jobs")
        job = random.choice(list_queue)
        mongo.declare_session(job)
        connect_to_queue(job)
        #consume_with_retry(job)
        xlsconnector = XlsConnector()
        if "ftp@" not in email:
            if not mongo.is_email_sent(job):
                mongo.close_session(job)
                csv_name = xlsconnector.process(queue=job)
                mail(username, recipient, password, "Nous avons le plaisir de vous informer que le job est terminé", "tmp.xlsx")
                xlsconnector.close()
                xlsconnector.clean()
                print("JOBS DONE FOR {}".format(job))

