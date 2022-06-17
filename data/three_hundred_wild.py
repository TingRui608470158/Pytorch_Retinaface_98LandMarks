import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import xml.etree.ElementTree as ET

class  three_hundred_wildDetection(data.Dataset):
    def __init__(self, data_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        
        tree = ET.parse(data_path)
        root = tree.getroot()
        labels = []
        for image_file in root[2]: #root[2]= dataset的image屬性
            for box_top in image_file: #image_file = tag:image; attrib:file,width,height
                #左上角xy坐标以及宽高
                top = int(box_top.attrib['top'])
                left = int(box_top.attrib['left'])
                width = int(box_top.attrib['width'])
                height = int(box_top.attrib['height'])
                bounding_box = top, left, width, height
                landmark_68 = []
                for landmark in box_top:
                    landmark_68.append(landmark.attrib['x'])
                    landmark_68.append(landmark.attrib['y'])
                label = landmark_68 + list(bounding_box)
                # print(label,len(label))
            labels.append(label)
            imgname = image_file.attrib['file']
            path = data_path.replace('labels_ibug_300W_train.xml',imgname) 
            labels.append(label)
            self.imgs_path.append(path)
        self.words = labels
        # print(self.imgs_path)
        # print(self.words)
        
    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        # print("imgs_path[index]=",self.imgs_path[index])
        height, width, _ = img.shape
        # print(len(self.words))
        label = self.words[index]
        annotations = np.zeros((0, 141))
        # print(annotations)
        if len(label) == 0:
            return annotations
            # print("\n\nlabel=",label)
            # print("len(label)=",len(label))
        # for idx, labe in enumerate(label):
        #     print("idx:",idx,labe)
        annotation = np.zeros((1, 141))
        # bbox
        annotation[0, 0] = label[-4]  # x1
        annotation[0, 1] = label[-3]  # y1
        annotation[0, 2] = label[-2]  # x2
        annotation[0, 3] = label[-1]  # y2

        # landmarks
        for nx in range(136):
            annotation[0, 4+nx] = label[nx]

        if (annotation[0, 4]<0):
            annotation[0, 140] = -1
        else:
            annotation[0, 140] = 1

        annotations = np.append(annotations, annotation, axis=0)
        # print(annotations)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

if __name__ == '__main__':
    data_path="./ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml"
    imgs_path = []
    labels = []
    tree = ET.parse(data_path)
    root = tree.getroot()
    for image_file in root[2]: #root[2]= dataset的image屬性
        for box_top in image_file: #image_file = tag:image; attrib:file,width,height
            #左上角xy坐标以及宽高
            top = int(box_top.attrib['top'])
            left = int(box_top.attrib['left'])
            width = int(box_top.attrib['width'])
            height = int(box_top.attrib['height'])
            bounding_box = top, left, width, height
            landmark_68 = []
            for landmark in box_top:
                landmark_68.append(landmark.attrib['x'])
                landmark_68.append(landmark.attrib['y'])
            label = list(bounding_box) + landmark_68
            # print(label,len(label))
        labels.append(label)

        imgname = image_file.attrib['file']
        path = data_path.replace('labels_ibug_300W_train.xml',imgname) 
        print(path)