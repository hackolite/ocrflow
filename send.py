from email.mime.text import MIMEText
import smtplib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def mail(recipient_email, sender_email, password, body, fichier_piece_jointe):

    # Paramètres du serveur SMTP Gmail
    smtp_server = 'smtp-mail.outlook.com'
    smtp_port = 587  # Port TLS

    # Votre adresse e-mail Gmail et mot de passe
    # Création de l'e-mail

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = 'XRETAIL'

    # Attacher le fichier
    with open(fichier_piece_jointe, 'rb') as piece_jointe:
        mime_part = MIMEBase('application', 'octet-stream')
        mime_part.set_payload(piece_jointe.read())
        encoders.encode_base64(mime_part)
        mime_part.add_header('Content-Disposition', f'attachment; filename={fichier_piece_jointe}')
        msg.attach(mime_part)

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