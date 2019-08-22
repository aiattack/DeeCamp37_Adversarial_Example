from utils import *
import sys
print(sys.path)
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model_irse import IR_50
import torch
import random
import math
from align_trans import get_reference_facial_points, warp_and_crop_face
from detector import detect_faces


# 读取数据集的文件列表
def get_imgs_path(dir_name):
    imgs_path = []
    imgs_dis = []
    g = os.walk(dir_name)
    for i, (path, dir_list, file_list) in enumerate(g):
        if i == 0:
            people = dir_list
        imgs_dis.append(len(file_list))
        for file_name in file_list:
            imgs_path.append(os.path.join(path, file_name))
    imgs_dis = imgs_dis[1:]        
    return imgs_path, imgs_dis, people


# 对齐人脸
def align_face(img_path, resize_face_size=112):
    img = Image.open(img_path)
    bounding_boxes, landmarks = detect_faces(img)
    scale = resize_face_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale
    facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(resize_face_size, resize_face_size))
    return warped_face


# 获得裁剪的人脸
def get_crop_face_file(crop_root, gen_root, gen=False):
    crop_paths = []
    gen_paths = []
    if os.path.exists(crop_root) is False:
        os.mkdir(crop_root)
    if os.path.exists(gen_root) is False:
        os.mkdir(gen_root)
    imgs_path,imgs_dis,people = get_imgs_path(data_root)  
    for img_path in imgs_path:
        temp = img_path.split("/")
        file_name = temp[-1]
        person =temp[-2]
        if gen:
            face = align_face(img_path)
            if os.path.exists(os.path.join(crop_root, person)) is False:
                os.mkdir(os.path.join(crop_root, person))
            if os.path.exists(os.path.join(gen_root, person)) is False:
                os.mkdir(os.path.join(gen_root, person))
            Image.fromarray(face).save(os.path.join(crop_root, person, file_name.replace('jpg', 'png')))
            Image.fromarray(face).save(os.path.join(gen_root, person, file_name.replace('jpg', 'png')))
        crop_paths.append(os.path.join(crop_root, person, file_name.replace('jpg', 'png')))
        gen_paths.append(os.path.join(gen_root, person, file_name.replace('jpg', 'png')))
    return crop_paths,gen_paths,imgs_dis,people