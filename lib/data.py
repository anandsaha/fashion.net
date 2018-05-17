"""Reads the DeepFashion dataset and provides a python interface to the same.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepFashion:

    def __init__(self, dataset_path):

        # The constants
        img_folder_name = "img"
        eval_folder_name = "Eval"
        anno_folder_name = "Anno"
        list_eval_partition_file = "list_eval_partition.txt"
        list_attr_img_file = "list_attr_img.txt"
        list_category_img_file = "list_category_img.txt"

        # The data structures

        # Each element is a tuple of (image path, category, attributes)
        self.train_imgs = []  # for all the training images
        self.test_imgs = []   # for all the test images
        self.val_imgs = []    # for all the validation images

        # Construct the paths
        self.path = dataset_path
        self.img_dir = os.path.join(self.path, img_folder_name)
        self.eval_dir = os.path.join(self.path, eval_folder_name)
        self.anno_dir = os.path.join(self.path, anno_folder_name)

        self.list_eval_partition = os.path.join(self.eval_dir, list_eval_partition_file)
        self.list_attr_img = os.path.join(self.anno_dir, list_attr_img_file)
        self.list_category_img = os.path.join(self.anno_dir, list_category_img_file)

        # Gather the Train, Test and Val image paths
        self.read_img_files_list()

    def read_img_files_list(self):

        fashion_db = "fashion.db"
        # If we already have the data structures on filesystem, read it and return
        if os.path.exists(fashion_db):
            print("Reading data structures from: ", fashion_db)
            db = open(fashion_db, "rb")
            self.train_imgs, self.val_imgs, self.test_imgs = pickle.load(db)
            print("Training images", len(self.train_imgs))
            print("Validation images", len(self.val_imgs))
            print("Test images", len(self.test_imgs))
            return

        # Read in the image to category mapping
        image_to_category = {}
        with open(self.list_category_img) as f:
            imgs_count = int(f.readline().strip())
            _ = f.readline().strip()  # read and throw away the header

            for line in f:
                words = line.split()
                image_to_category[words[0].strip()] = int(words[1].strip())
        assert(imgs_count == len(image_to_category))

        # Read in the image to attributes mapping
        image_to_attributes = {}
        with open(self.list_attr_img) as f:
            imgs_count = int(f.readline().strip())
            _ = f.readline().strip()  # read and throw away the header
            for line in f:
                words = line.split(sep='jpg')
                lst = [int(i) for i in words[1].strip().split()]
                image_to_attributes[words[0].strip()+"jpg"] = lst
        assert(imgs_count == len(image_to_attributes))

        # Read in the images
        with open(self.list_eval_partition) as f:
            imgs_count = int(f.readline().strip())
            _ = f.readline().strip()  # read and throw away the header

            for line in f:
                words = line.split()
                img = words[0].strip()
                category_idx = image_to_category[img]
                category = np.zeros(50)  # one hot encoded
                category[category_idx - 1] = 1
                attributes = image_to_attributes[img]

                if words[1].strip() == "train":
                    self.train_imgs.append((img, category, attributes))
                if words[1].strip() == "val":
                    self.val_imgs.append((img, category, attributes))
                if words[1].strip() == "test":
                    self.test_imgs.append((img, category, attributes))

        print("Training images", len(self.train_imgs))
        print("Validation images", len(self.val_imgs))
        print("Test images", len(self.test_imgs))
        assert(imgs_count == (len(self.train_imgs)+len(self.test_imgs)+len(self.val_imgs)))

        # Store the data structures
        db = open(fashion_db, "wb")
        pickle.dump((self.train_imgs, self.val_imgs, self.test_imgs), db)
        db.close()
        print("Data structures stored on filesystem as: ", fashion_db)


#df = DeepFashion("/home/as/datasets/lily/deep-fashion")
#print(df.val_imgs[0])
