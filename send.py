from email.mime.text import MIMEText
import smtplib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


#subject = "XRETAIL JOBS"
#body = "LE JOB  A ETE ENVOYEE"
#password = ""
#sender_email = 'laureote-loic@hotmail.fr'
# Adresse e-mail du destinataire
#recipient_email = 'loic.laureote@gmail.com'


def mail(recipient_email, sender_email, password, body):

    # Paramètres du serveur SMTP Gmail
    smtp_server = 'smtp-mail.outlook.com'
    smtp_port = 587  # Port TLS

    # Votre adresse e-mail Gmail et mot de passe
    # Création de l'e-mail

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = 'XRETAIL'

    msg.attach(MIMEText(body, 'plain'))
    # Connexion au serveur SMTP
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    # Authentification
    server.login(sender_email, password)
    # Envoi de l'e-mail
    server.sendmail(sender_email, recipient_email, msg.as_string())
    # Fermeture de la connexion SMTP
    server.quit()
    print("E-mail envoyé avec succès !")


#mail(recipient_email, sender_email, password, body)