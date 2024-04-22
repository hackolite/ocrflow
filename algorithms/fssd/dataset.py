import os


class Dataset:

    @staticmethod
    def filter_images(folder_images=None, folder_labels=None, method="del_empty"):
        data = os.listdir(images_dir)
        for sample in data:
            name = sample.replace(".jpg","").replace(".jpeg","")
            path_xml = os.path.join(folder_labels=, name+".xml")
            path_img = os.path.join(folder_images, sample)

            with open(path_xml) as fd:
                try:
                    doc = xmltodict.parse(fd.read())
                    _ = doc["annotation"]["object"]

                except:
                    os.remove(path_xml)
                    os.remove(path_img)


    @staticmethod
    def pascalvoc_icdar(folder_images=None, folder_labels=None, method="del_empty"):
        pass


    @staticmethod
    def pascalvoc_yolo(folder_images=None, folder_labels=None, method="del_empty"):
        pass

    @staticmethod
    def pascalvoc_h5(folder_images=None, folder_labels=None, method="del_empty"):
        pass


    @staticmethod
    def craft_pascalvoc():
        pass
