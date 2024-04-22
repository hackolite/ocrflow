from email.mime.text import MIMEText
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase  # Importer MIMEBase
from email import encoders  # Importer encoders

def mail(recipient_email, sender_email, password, body):

    # Paramètres du serveur SMTP Gmail
    smtp_server = 'smtp-mail.outlook.com'
    smtp_port = 587  # Port TLS

    # Votre adresse e-mail Gmail et mot de passe
    # Création de l'e-mail

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = 'XRETAIL'
    message.attach(MIMEText(body, 'plain'))

    # Attachement du fichier
    filename = 'tmp.xlsx'
    attachment = open(filename, 'rb')
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    message.attach(part)

    # Connexion au serveur SMTP et envoi de l'email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, password)
        text = message.as_string()
        server.sendmail(sender_email, recipient_email, text)
        print("E-mail envoyé avec succès !")

