import cv2
import keras_ocr
import os 
import time


start = time.time()

im_fl = os.listdir("./pad")


def create_images(im_fl=None):
    for i in im_fl:
        if 'jpg' in i:
            detector = keras_ocr.detection.Detector()
            detector.model.load_weights("detector_born_digital.h5")
            image = keras_ocr.tools.read("./pad/"+i)
            image = cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
            boxes = detector.detect(images=[image])[0]
        
            for box in boxes:
                try:
                    xmin, ymin, xmax, ymax = int(box[0][0]) ,int(box[0][1]), int(box[2][0]), int(box[2][1])
                except Exception as e:
                    print(e)
                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),7)
            cv2.imwrite("test_"+i, image)


def PASCALVOC2ICDAR2015():
	pass

