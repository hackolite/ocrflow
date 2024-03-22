#!/bin/bash
source /home/ubuntu/ocrflow/ocrflow/infrastructure/env
# Spécifiez le répertoire à surveiller
directory=$FTP_FOLDER
# Utilisez inotifywait pour surveiller le répertoire en continu
inotifywait -m -e close_write "$directory" |
while read -r path event file; do
    echo "Le fichier $file a été téléchargé. Exécution de l'action..."
    python3.8 cloudify.py --name $CLOUDINARY_NAME --key $CLOUDINARY_KEY --secret $CLOUDINARY_SECRET --path $FTP_FOLDER --hostname $API_HOST
    # Ajoutez votre commande ou script ici
done
