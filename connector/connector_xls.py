# Set the dimensions of the cell where you want to insert the image
from connector.connector_mongodb import MongoConnector
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
import random
import string
import cv2
import numpy as np
import xlsxwriter 
import os
from configuration import *





class XlsConnector:

    def __init__(self, user="xretail", image_path=".", excel_path="tmp.xlsx", session=None):

        self.workbook = xlsxwriter.Workbook(excel_path)
        self.worksheet = self.workbook.add_worksheet()
        bool_format = self.workbook.add_format({'bold': True, 'color': 'red'})
        # Create a format for the specific cell
        #cell_format = workbook.add_format({'bold': True, 'color': 'red'})
        # Write a blank cell with the specified format to the specific cell ('A1')

        ROW_IDX = 1
        #self.worksheet.set_default_row(200)
        #self.worksheet.set_column('A:A', None, bool_format)
        
        # Définir la largeur de la colonne A en fonction de la largeur de l'image
        self.worksheet.set_column('A:A', 384 / 3)  # Diviser par 7 pour convertir les pixels en points Excel
        # Définir la hauteur de la ligne 1 en fonction de la hauteur de l'image
        self.worksheet.set_default_row(512 / 3)  # Diviser par 15 pour convertir les pixels en points Excel


        self.worksheet.set_column('B:H', 20)
        self.worksheet.set_column('F:F', 80)
        self.worksheet.set_column('A:A', 100)
        self.worksheet.write('A1', 'image')
        self.worksheet.write('B1', 'gtin')
        self.worksheet.write('C1', 'price')
        self.worksheet.write('D1', 'd_check')
        self.worksheet.write('E1', 'h_check')
        self.worksheet.write('F1', 'link')
        self.worksheet.write('G1', 'osm_id')
        self.worksheet.write('H1', 'file')



    def add(self, value=None,index=None):
        print("image :", value[0], index+1)
        self.worksheet.insert_image('A{}'.format(index+1), value[0], {'x_scale': 1, 'y_scale': 1})
        self.worksheet.write_row(index, 1, value[1:])


    def close(self):
        self.workbook.close()

    def execute(self):
        return excel_path  



    def process(self, queue=None):
        mongoconnector = MongoConnector(user=mongodb_credentials["user"] , password=mongodb_credentials["password"], host=mongodb_credentials["host"])
        self.queue = queue
        resultats = mongoconnector.get_queue_session(queue=queue)
        for index, i in enumerate(resultats):
            index +=1
            print(i["annotation"]['position'], i["ean"], i["price"], i['valid'], i['valid'], i["url_image"], i["store_id"])
            raw_image = get_image(i["url_image"])
            image = crop_image(raw_image, i["annotation"]['position'])
            #print("SIZE :", image.shape)
            name = "{}.jpeg".format(get_random_string(10))
            
            try:
                image = cv2.resize(image, (600,200), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(name, image)
            
            except Exception as e:
                print(e)
                raw_image = cv2.resize(raw_image, (600,200), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(name, raw_image)
            

            value = [name, i["ean"], i["price"], i['valid'], i['valid'], i["url_image"], i["store_id"], name]    
            self.add(value, index)
        self.close()
        return name

    def clean(self):
        os.system("mkdir {}".format(self.queue))
        os.system("mv *.jpeg {}".format(self.queue))
        os.system("mv  tmp.xlsx {}".format(self.queue))




def get_image(url_im):
    response = urlopen(url_im)
    image_data = response.read()
    image = Image.open(BytesIO(image_data))
    largeur, hauteur = image.size
    print("SIZE :", largeur, hauteur)
    img = np.array(image)
    
    return img

def crop_image(image, box):
    pad = image[box[1]:box[3], box[0]:box[2]]
    return pad



def get_random_string(length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))




#doplqsxlch


