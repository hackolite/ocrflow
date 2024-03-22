# app.py
from flask import Flask, request, jsonify
import pika
import json 
import time
import os 

app = Flask(__name__)

hostname = os.getenv('RABBITMQ_HOST')
user = os.getenv('RABBITMQ_DEFAULT_USER')
password = os.getenv('RABBITMQ_DEFAULT_PASS')


time.sleep(5)
credentials = pika.PlainCredentials(user, password)
parameters = pika.ConnectionParameters(hostname,5672,'/',credentials)


@app.route('/')
def server():
    # Connection RabbitMQ
    return 'Job sent to RabbitMQ!'


@app.route('/ocr/full', methods=['POST'])
def receive_post():
    if request.method == 'POST':
        # Connection RabbitMQ
        data = request.json  # Récupère les données JSON de la requête
        if data is not None and 'session' in data:
            nom_queue = data['session']
            print("Nom de la queue:", nom_queue)
        else:
            print("Paramètre 'session' non trouvé dans la requête JSON.")

        with pika.BlockingConnection(parameters) as connection:
            channel = connection.channel()
            # Déclaration de la file
            try:
                # Essayer de déclarer la queue en mode passif
                channel.queue_declare(queue=nom_queue, auto_delete=True)
                print("La queue existe déjà.")
            except pika.exceptions.ChannelClosedByBroker as e:
                pass
            channel.basic_publish(exchange='',routing_key=nom_queue,body=request.get_data())
        return jsonify(request.json)


@app.route("/ping", methods=['GET'])
def status(): 
    return {"status":"OK"}



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
