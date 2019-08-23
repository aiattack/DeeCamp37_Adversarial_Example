from align import detector
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_images(imgs):
    for i,img in enumerate(imgs):        
        plt.subplot(2,2,i+1)
        plt.imshow(imgs[i])

def align_face(img,size=(112,112)):
    reference = get_reference_facial_points(default_square = True) * size[0]/112.
    landmarks= detector.detect_faces(img)[1]
    facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, size)
    return warped_face

def img2tensor(img):
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    img = np.reshape(img, [1, 3, 112, 112])
    img = np.array(img, dtype = np.float32)    
    img = (img - 127.5) / 128.0    
    img = torch.from_numpy(img)
    return img

def l2_norm(input, axis = 1):    
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output