folder_annotations = "./annotations"
folder_pascalvoc = "/home/lamaaz/XRETAIL_INFRASTRUCTURE/craft/pad"
import os
import xmltodict

annotations = os.listdir(folder_pascalvoc)
for files in annotations:
	
    if ".xml" in files:
        #xml_file = open(files.replace(".jpg", ".xml"), "wb")
        with open(folder_pascalvoc +"/"+ files) as fd:
            gt_file = open("gt_"+files.replace(".xml", ".txt"), "w")
            doc = xmltodict.parse(fd.read())
            try:
                print(doc["annotation"]["object"])	
                #gt_file = open("gt_"+files.replace(".jpg", ".xml"), "w")
    
                if type(doc["annotation"]["object"]) == dict:
                    coordinate = doc["annotation"]["object"]['bndbox']
                    lst_coordinate = [coordinate["xmin"], coordinate["ymin"],  coordinate["xmax"], coordinate["ymin"],  coordinate["xmax"], coordinate["ymax"],  coordinate["xmin"], coordinate["ymax"], "text"] 
                    gt_file.write(",".join(str(coord) for coord in lst_coordinate)) 
                    

                elif type(doc["annotation"]["object"]) == list:
                    for box in doc["annotation"]["object"]:
                        coordinate = box['bndbox']
                        lst_coordinate = [coordinate["xmin"], coordinate["ymin"],  coordinate["xmax"], coordinate["ymin"],  coordinate["xmax"], coordinate["ymax"],  coordinate["xmin"], coordinate["ymax"], "text"] 
                        line = ",".join(str(coord) for coord in lst_coordinate)
                        gt_file.write(line + "\n") 
                
                gt_file.close()
            
            except Exception as e:
                print(e)
        
            
