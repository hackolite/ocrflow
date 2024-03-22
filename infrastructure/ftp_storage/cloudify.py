import os
import zipfile
import random
import string
import shutil
import cloudinary
import requests
from cloudinary import uploader
from configuration import cloudinary_credentials, api 
import argparse


API_ENDPOINT = "http://{}/ocr/full"


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def remove_folder(directory_path):
    try:
        shutil.rmtree("/".join(directory_path.split("/")[:-1]))
        os.remove(directory_path[:-1]+".zip")
        print(f"Folder '{directory_path}' successfully removed.")
    except Exception as e:
        print(f"Error removing folder '{directory_path}': {e}")


def generate_random_string(length=12):
    characters = string.ascii_letters
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


def extract_images(path=None):
	"""extract images from zip file"""
	random_path = generate_random_string(6)
	for archive in os.listdir(path):
		if ".zip" in archive :
			print(archive)
			with zipfile.ZipFile("{}/{}".format(path,archive), 'r') as zip_ref:
				zip_ref.extractall("{}".format(path))
				folder_origin =  zip_ref.namelist()[0]
			os.replace("{}/{}/".format(path, folder_origin), "{}/{}/".format(path, archive.replace(".zip","")))

		else:
			upload_file = "{}/{}".format(path,archive)
			os.remove(upload_file)
			return None

	return "{}/{}/".format(path, archive.replace(".zip",""))


def push_images(path=None, name=None, key=None, secret=None):
	""" push images to cloudinary"""
	cloudinary.config(
	cloud_name = name,
	api_key = key,
	api_secret = secret
	)
	pushed_images = []
	for i in os.listdir(path):
		if ".jpeg" or ".png" or ".jpg" in path:
			resp = uploader.upload("{}{}".format(path,i))
			pushed_images.append(resp["url"])	
	return pushed_images		



def send_api(list_images=None, shop_id=None, user_id=None, hostname=None):
	"""send images to pprocessing"""
	session = get_random_string(10)
	for i in  list_images:
            # Données à envoyer dans la requête POST (au format JSON)  
            data = {'shop_id': shop_id, 'url_image': i,  "user_id": user_id, "session" : session}
            response = requests.post(API_ENDPOINT.format(hostname), json=data)
            print(i, response)


def clean(path):
        remove_folder(path)


def execute(path, name=None, key=None, secret=None, hostname=None):
        folder = extract_images(path)
        session = folder.split("/")[-2]
        try:
            user_id, shop_id = session.replace(".zip","").split('+')
        except:
            user_id, shop_id = "ftp@ftp.com", session.replace(".zip","")

        if folder :
        	images_list = push_images(folder, name=name, key=key, secret=secret)
        	send_api(list_images=images_list, shop_id=shop_id,  user_id=user_id, hostname=hostname)
        	clean(folder)


if __name__ == '__main__':
       parser = argparse.ArgumentParser(description='Process images from a specified path')
       # Add path argument
       parser.add_argument('--path',   metavar='PATH', type=str, help='image path')
       parser.add_argument('--name',   metavar='NAME', type=str, help='cloudinary name')
       parser.add_argument('--key',    metavar='KEY', type=str, help='cloudinary key')
       parser.add_argument('--secret', metavar='SECRET', type=str, help='cloudinary secret')
       parser.add_argument('--hostname', metavar='HOSTNAME', type=str, help='hostname')
       # Parse arguments
       args = parser.parse_args()
       # Execute with the specified path
       execute(args.path, name=args.name, key=args.key, secret=args.secret, hostname=args.hostname)
       #execute("/home/ubuntu/xretail_plateform/xretail_front/app/forms/uploads")
