import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
import argparse
from util.img_util import plot_images, align_face, img2tensor, l2_norm
from backbone.model_irse import IR_50
from face_util import face_recognition

IMG_SIZE = 224

model = IR_50([112,112])
model.load_state_dict(torch.load('backbone_ir50_ms1m_epoch120.pth',map_location='cpu'))
model.eval()
criterion = nn.MSELoss()
if torch.cuda.is_available() == False:
    device = 'cpu'
else: 
    device = 'cuda'

# collect all images to create corresponding adversarial examples
# './data/adversarial' is the container of adversarial face images
paths = []
advpath = './data/adversarial/'
dire = None

for root, dirs, files in os.walk('./data/test'): 
    if dirs:
        dire = dirs
    for f in files:
        paths.append(os.path.join(root, f))
for _d in dire:        
    os.system('mkdir -p ' + advpath + _d) 

eps = 1 
steps = 8

nums_img = len(paths)
counter = 0

for path in paths:
    print('processing ' + path)
    # gt: the predicted person, considered as gound_truth     target: the 2nd closest person feature    
    # in_tensor: the input tensor of input image
    # gt_feat: the groundtruth tensor     target_feat: the target people tensor    
    # gt_confi: groundtruth confidence       target_confi: the target peoples confidence    
    (gt, target), in_tensor, (gt_feat, target_feat), (gt_confi, target_confi) = face_recognition(path, isInit=True)
    print(gt + ' ' + target)

    in_variable = Variable(in_tensor, requires_grad=True) 
    gt_feat = gt_feat.reshape((1, 512))
    gt_feat = gt_feat.astype(np.float32)
    gt_feat = torch.from_numpy(gt_feat)

    target_feat = target_feat.reshape((1, 512))
    target_feat = target_feat.astype(np.float32)
    target_feat = torch.from_numpy(target_feat)

    in_tensor = in_tensor.squeeze()

    adv = None
    perturbation = None
    name = None
    out_feat = None

    for i in range(steps):
        print('step: '+str(i))
        out_feat = model(in_variable)        
        (fir_name, sec_name), _, (fir_confi, sec_confi) = face_recognition(out_feat, isTensor=True)
        print(fir_name + '  ' + str(fir_confi))
        print(sec_name + '  ' + str(sec_confi))           

        loss = criterion(out_feat, gt_feat) - criterion(out_feat, target_feat)
        print('the loss is: ' + str(loss.item()))
        # compute gradients
        loss.backward()

        in_variable.data = in_variable.data + ((eps/255.) * torch.sign(in_variable.grad.data))
        in_variable.grad.data.zero_() # unnecessary

        # deprocess image
        adv = in_variable.data.cpu().numpy()[0] # (3, 112, 112)   
        perturbation = (adv - in_tensor.numpy())

    adv = adv*128.0 + 127.0
    adv = adv.swapaxes(0,1).swapaxes(1,2)
    adv = adv[...,::-1]
    adv = np.clip(adv, 0, 255).astype(np.uint8)
    idx = path.split('/')[-1].split('.')[-2]
    (name, sename), _, _ = face_recognition(out_feat, isTensor=True)
    if name != gt: 
        # attack successfully!
        counter += 1
        advimg = advpath+'/'+gt+'/'+name+idx+'.jpg'
    else:
        advimg = advpath + '/'+gt+'/'+gt+idx+'.jpg'
    cv2.imwrite(advimg, adv)    

print('the error rate is: ', counter/nums_img)