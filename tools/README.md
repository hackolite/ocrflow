# Plateforme de MLOps pour le déploiement de modèles YOLO

## Introduction

Cette plateforme de MLOps est conçue pour déployer et gérer des modèles de détection d'objet et de détection et de recognition de texte tel YOLO (You Only Look Once), TrOcr, SSD. Elle gère un flux de données où les images sont déposées dans un serveur FTP, stockées dans un service Cloudinary, les liens de stockage sont envoyés à une API Flask pour le traitement, un système de messagerie RabbitMQ est utilisé pour la mise en file d'attente et le traitement asynchrone des images, et finalement, les résultats sont stockés dans une base de données MongoDB.

## Fonctionnalités

- Déploiement et gestion d'architecture OCR 3 étapes crop/detection/recognition.
- Traitement asynchrone des images via RabbitMQ.
- Stockage des images sur Cloudinary et des résultats dans MongoDB.
- Gestion centralisée via une API Flask.

## Architecture

![Architecture de la plateforme](architecture.png)

## Installation

### Prérequis

- Docker et Docker Compose installés sur votre machine.
- Accès à un serveur FTP.
- Clé API Cloudinary.
- Serveur RabbitMQ.
- Serveur MongoDB.

### Étapes d'installation

1. Clonez ce dépôt sur votre machine :

git clone https://github.com/hackolite/ocrflow.git

2. Créez un fichier `.env` à la racine du projet et définissez les variables d'environnement requises :
```dotenv
MODEL_FOLDER           =  /home/lamaaz/xretail_plateform/ocrflow/models
API_HOST               =  host

FTP_HOST               =  host
FTP_USER               =  user
FTP_PASSWORD           =  password

CLOUDINARY_NAME        =  test
CLOUDINARY_KEY         =  test
CLOUDINARY_SECRET      =  secret

RABBITMQ_HOST          =  cluster0.nn3l2bm.mongodb.net
RABBITMQ_USER          =  xretail
RABBITMQ_PASSWORD      =  password

MONGODB_HOST           =  host
MONGODB_USER           =  user
MONGODB_PASSWORD       =  password

OPENFOODFACT_USER      = user
OPENFOODFACT_PASSWORD  = password


Lancez la plateforme en utilisant Docker Compose dans chaque dossier dans /ocrflow/infrastructure.
On y trouve trouve apiQ et ftp_storage
On lance avec avec :

cd /ocrflow/infrastructure
Executer :

1. cd /ocrflow/infrastructure/ftp_storage/
1. ./ocrflow/infrastructure/ftp_storage/start.sh #lancement de ftp
2. cd ../apiQ
3. docker-compose .env -d up #


## Utilisation

1. Uploader des images sur le serveur FTP.
2. Les images sont automatiquement téléchargées vers Cloudinary.
3. Les liens de stockage des images sont envoyés à l'API Flask.
4. L'API Flask publie les liens dans la file d'attente RabbitMQ.
5. Le consumer process traite les images avec le modèle YOLO.
6. Les résultats de détection sont stockés dans MongoDB.


## Consommer
1. Pour consommer, lancez python3.8 inf_consumer.py


## Contribuer

Les contributions sont les bienvenues ! Pour des modifications importantes, veuillez ouvrir d'abord une issue pour discuter de ce que vous aimeriez modifier.

## Licence

Ce projet est sous licence MIT
