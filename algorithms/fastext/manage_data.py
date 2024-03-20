import os
from sklearn.model_selection import train_test_split
import csv
import xmltodict



def rename_dataset(dataset_dir=False, train_pourcentage=None):
	"""convert pascalvoc roboflow dataset to pure pascalvoc dataset"""
	Annotations ='{0}/Annotations/'.format(dataset_dir)
	Images  ='{0}/Images/'.format(dataset_dir)
	files = os.listdir(Images)
	for file in files :
		if ".jpeg" in file:
			print(file)
			new_file = file.replace(".jpeg",".jpg")
			os.rename('{0}Images/{1}'.format(dataset_dir, file), '{0}Images/{1}'.format(dataset_dir, new_file))






def filter_dataset(dataset_dir=None):
	""" image and annotation presents """
	Annotations ='{0}/Annotations/'.format(dataset_dir)
	Images  ='{0}/Images/'.format(dataset_dir)
	
	Anns = os.listdir(Annotations)
	Imgs = os.listdir(Images)

	Anns = [an.replace(".xml", "") for an in Anns]
	Imgs = [im.replace(".jpg", "") for im in Imgs]
	
	set_anns = set(Anns)
	set_imgs = set(Imgs)
	
	resultats = set_anns.symmetric_difference(set_imgs)
	print(len(Anns))
	print(len(Imgs))
	for i in resultats:
		print(i)



def intersection_dataset(dataset_dir=None):
	""" image and annotation presents """
	Annotations ='{0}/Annotations/'.format(dataset_dir)
	Images  ='{0}/Images/'.format(dataset_dir)
	
	Anns = os.listdir(Annotations)
	Imgs = os.listdir(Images)

	Anns = [an.replace(".xml", "") for an in Anns]
	Imgs = [im.replace(".jpg", "") for im in Imgs]
	
	set_anns = set(Anns)
	set_imgs = set(Imgs)
	
	resultats = set_anns.intersection(set_imgs)
	
	return list(resultats)






def generate_training_files(dataset_dir=None, test_pourcentage=None):
	dataset_elements = intersection_dataset(dataset_dir="./dataset/PAD/")
	x_train ,x_test = train_test_split(dataset_elements,test_size=test_pourcentage) 
	empty_elements = malformed_annotations_dataset(annotations_dir="./dataset/PAD/")
	with open('trainval.txt', 'w', newline='') as file:
		writer = csv.writer(file)	    
		for i in x_train:
			if i not in empty_elements:
				writer.writerow([str(i)])

	with open('test.txt', 'w', newline='') as file:
		writer = csv.writer(file)
		for i in x_test:
			if i not in empty_elements:
				writer.writerow([str(i)])



def malformed_annotations_dataset(annotations_dir=None):
	malformed_id = []
	Annotations ='{0}/Annotations/'.format(annotations_dir)
	for xml_p in os.listdir(Annotations):
		try:
			xml_path = os.path.join(Annotations, xml_p)
			with open(xml_path) as fd:
				doc = xmltodict.parse(fd.read())
				obj = doc["annotation"]["object"]

		except Exception as  e:
			#print(e)
			#print(xml_p)
			malformed_id.append(xml_p.replace(".xml",""))
	return malformed_id


if __name__ == '__main__':
	#rename_dataset(dataset_dir="./dataset/PAD/", )
	#filter_dataset(dataset_dir="./dataset/PAD/", )
	#dataset_elements = intersection_dataset(dataset_dir="./dataset/PAD/")
	generate_training_files(dataset_dir="./dataset/PAD/",test_pourcentage=0.2)
	dataset_elements = malformed_annotations_dataset(annotations_dir="./dataset/PAD/")
	#print(dataset_elements)