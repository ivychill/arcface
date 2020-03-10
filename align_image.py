import cv2
import PIL
from PIL import Image
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
import numpy as np
import os
from pathlib import Path

if __name__ == '__main__':
    mtcnn = MTCNN()
    conf = get_config(False)
    path=r'/home/test/query_kc/'
    for root, dirs, files in os.walk(path):
        for file in files:
            image_path = root+file
            #img = Image.open(image_path)
            img = cv2.imread(image_path)
            img = Image.fromarray(img)
            #img = mtcnn.align(img)
            bboxes, faces = mtcnn.align_multi(img, conf.face_limit, conf.min_face_size) 
            #print(bboxes)
            #print(faces[0].size)
            #cv2.imwrite("")
            #inverted_image = Image.fromarray(faces[0][...,::-1]) #bgr to rgb
            #inverted_image.save("/home/test/"+"aline_image.jpg")
            write_root = root.replace('query_kc', 'aligned')
            if not os.path.exists(write_root):
                os.makedirs(write_root)
            cv2.imwrite(write_root+file, np.array(faces[0]))
            #cv2.imwrite("/home/test/"+"aline_image.jpg",np.array(faces[0]))
            #print(len(img))
            #cv2.imwrite("/home/test/"+"aline_image.jpg",np.array(img))
