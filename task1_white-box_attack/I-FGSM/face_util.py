import sys
sys.path.append('./align')
import os
import numpy as np
import cv2
import torch
import pickle as pkl
import matplotlib.pyplot as plt
from PIL import Image
from backbone.model_irse import IR_50
from Utils.img_util import plot_images, align_face, img2tensor, l2_norm

model = IR_50([112,112])
model.load_state_dict(torch.load('backbone_ir50_ms1m_epoch120.pth',map_location='cpu'))
model.eval()

# @function: store the target image data processed by the model
# @parameters: isAver: True: store the average face feature data  False: store each face feature data
# @attention: the face feature data isn't normalized
def save_imgdata(root_dir='./data/train/', isAver=False):
    paths = os.listdir(root_dir)
    if not isAver:
        warehouse = 'warehouse_3.pkl'
    else:
        warehouse = 'aver_warehouse.pkl'
    num_persons = len(paths)
    num_faces = len(os.listdir(root_dir+paths[0]))
    face_datas = np.zeros((512, num_faces*num_persons))        
    names = []
    ids = []

    i = 0

    for path in paths:
        name = path.split('/')[-1]        
        path = root_dir + path                        
        imgs_path = os.listdir(path)        
        for img_path in imgs_path:
            names.append(name)
            ids.append(i)
            img_path = path + '/' + img_path                                    
            img_align = align_face(Image.open(img_path))
            tmp = model(img2tensor(img_align))                        
            tmp = tmp.detach().numpy()            
            face_datas[:,i] = tmp # not normalized!
            i += 1        

    if isAver:
        aver_fds = np.zeros((512, num_persons))
        ids = [i for i in range(num_persons)]
        for i in range(num_persons):
            aver_fds[:,i] = np.sum(face_datas[:,i:i+num_faces]).reshape(512,1)
        face_datas = None
        face_datas = aver_fds

    face_info = dict(zip(ids, names))
    with open(warehouse, 'wb+') as file:
        pkl.dump((face_datas,face_info), file)
    file.close()        

# @function: due to some stupid operations, we need to get the average face feature data from each face feature
# @parameters: warehouse: stores each face feature data
def average_faces(warehouse='warehouse_3.pkl', num_persons=19, num_faces=20):
    with open(warehouse, 'rb+') as file:
        face_datas, face_info = pkl.load(file)        
    file.close()
    aver_warehouse = 'aver_'+warehouse

    aver_fds = np.zeros((512, num_persons))
    ids = [i for i in range(num_persons)]
    names = []
    for i in range(num_persons):
        names.append(face_info[i*num_faces])

    for i in range(num_persons):        
        aver_fds[:,i] = np.sum(face_datas[:,num_faces*i:num_faces*(i+1)], axis=1)/num_faces    

    face_info = dict(zip(ids, names))
    with open(aver_warehouse, 'wb+') as file:
        pkl.dump((aver_fds,face_info), file)

    file.close()

# @function: face recognition with input images or tensors
# @paremeters: 
#           img: image path or tensors      isInit: True: the input img is origin, without noise
#           isTensor: input img is tensor   aligned_path: stores the aligned face image           
# @output:
#           person_name: the closest person     seperson_name: the 2nd closest person          
def face_recognition(img, warehouse='aver_warehouse_3.pkl', isInit=False, isTensor=False, aligned_path='./data/origin/'):
    with open(warehouse, 'rb+') as file:
        face_datas, face_info = pkl.load(file)        
    file.close()            

    if isTensor:   
        feat_img = img
    else:         
        imgpath = img.split('/')[-2] + img.split('/')[-1]        
        img = Image.open(img)
        img_aligned = align_face(img)                
        origin_path = aligned_path+imgpath        
        cv2.imwrite(origin_path, img_aligned[...,::-1])
        feat_img = model(img2tensor(img_aligned))

    feat_imgnp = l2_norm(feat_img)
    feat_imgnp = feat_imgnp.detach().numpy()
    max_sim = 0
    pred = ''
    pred_img = None        

    # find the closest person
    norm_faces = face_datas / np.linalg.norm(face_datas, axis=0)    
    similaritis = feat_imgnp.dot(norm_faces)        
    confidence = np.max(similaritis, axis=1)
    person_id = np.argmax(similaritis, axis=1)    
    person_name = face_info[person_id.item()]
    person_feat = face_datas[:,person_id] # person_feature hasn't already normalized!!!

    # find the 2nd closest person
    import copy
    _similar = copy.deepcopy(similaritis)
    _similar[0,person_id.item()] = np.min(_similar)
    seconfi = np.max(_similar, axis=1)
    seperson_id = np.argmax(_similar, axis=1)
    seperson_name = face_info[seperson_id.item()]
    seperson_feat = face_datas[:,seperson_id]       

    if isInit:
        return (person_name, seperson_name), img2tensor(img_aligned), (person_feat, seperson_feat), (confidence, seconfi)
    return (person_name, seperson_name), (person_feat, seperson_feat), (confidence, seconfi)
