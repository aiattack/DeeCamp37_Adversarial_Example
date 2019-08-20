'''
目标攻击
思路：
将目标与本人的若干张图片、若干模型载入指定设备，
指定一张图片（a）生成它的对抗样本，
不断改变模型和目标人脸图片(b)，与（a）为同一人的图片（c);
使用梯度下降法使得a,b的相似度变高;a与c的相似度变低。'''

import time
t1 = time.time()


import random
import torch
from utils import *
from utils import utils
from utils import get_face
from model_irse import IR_50, IR_101, IR_152
from model_resnet import ResNet_50, ResNet_101
from utils.guass import *
import multiprocessing
from utils import config


# 二阶tv loss
def tv_loss(input_t):
    temp1 = torch.cat((input_t[:,:,1:,:], input_t[:,:,-1,:].unsqueeze(2)), 2)
    temp2 = torch.cat((input_t[:,:,:,1:], input_t[:,:,:, -1].unsqueeze(3)), 3)
    temp = (input_t - temp1)**2 +  (input_t - temp2)**2
    return temp.sum()


# 初始化干扰噪声
def get_init_noise(device):
    noise = torch.Tensor(1, 3, 112, 112)
    noise = torch.nn.init.xavier_normal(noise, gain=5)
    return noise.to(device)
    #print(noise.abs().mean())


# 更换模型  
def change_model(model_pool):
    s = len(model_pool)
    index = random.randint(0, s - 1)
    return model_pool[index]


# 更换v2, v2是目标人脸的特征向量 512维
def change_v2(crop_path_pool, model, device):
    s = len(crop_path_pool)
    index_target = random.randint(0, s - 1)
    print('change v2, v2 index is %d' % (index_target, ))
    target = crop_path_pool[index_target]
    v2 = l2_norm(model(target))
    v2 = v2.detach_()
    return v2


# 单次迭代
def iter_noise(noise, img_origin, gaussian_blur, model, v2, lr=1, is_v2_self=False):
    noise.requires_grad = True
    noise1 = gaussian_blur(noise)
    img_origin = img_origin.detach_()
    v1 = l2_norm(model(img_origin + noise1))
    if is_v2_self:
        loss1 = (v1*v2).sum()
    else:
        loss1 = 1 - (v1*v2).sum()
    loss2 = (noise1**2).sum().sqrt()
    loss3 = tv_loss(noise1)
    loss = loss1 + 0.0025 * loss2 + 0.004 * loss3
    loss.backward()
    print(loss1.item(), loss2.item() * 128 / 112, loss3.item(), noise.grad.abs().sum())
    

    if is_v2_self:
        lr = 0.5 * lr
    if loss1 < 0.5:
        lr = 0.5 * lr
    if loss1 < 0.1:
        lr = 0.5 * lr    
    noise = noise.detach() - lr * noise.grad.detach()
    noise = (noise + img_origin).clamp_(-1, 1) - img_origin
    noise = noise.clamp_(-0.2, 0.2)
    return noise


# 多次迭代
# m 几次换模型， n 基础换目标样本
def get_noise(model_pool, img_origin, gaussian_blur, target_face_pool, same_person_pool, n, m, device):
    noise = get_init_noise(device)
    is_v2_self = False
    i = 0
    while True:
        if i % m == 0:
            model = change_model(model_pool)
        if i % n == 0:
            if is_v2_self:
                v2 = change_v2(target_face_pool, model, device)
                is_v2_self = False
            else:
                v2 = change_v2(same_person_pool, model, device)
                is_v2_self = True
        noise = iter_noise(noise, img_origin, gaussian_blur, model, v2, lr=2, is_v2_self=is_v2_self)
        yield noise
        i += 1 


# 初始化模型
def init_model(model, param, device):
    m = model([112,112])
    m.eval()
    m.to(device)
    m.load_state_dict(torch.load(param))
    return m


# 初始化图片池
def get_img_pool(person_list, device):
    person_pool = []
    for el in person_list:
        person_pool.append(utils.to_torch_tensor(Image.open(el)).unsqueeze_(0).to(device))
    return person_pool


#初始化模型池
def get_model_pool(device):
    model_pool = []   
    model_pool.append(init_model(IR_50, 'models/backbone_ir50_ms1m_epoch120.pth', device))
    model_pool.append(init_model(IR_101, 'models/Backbone_IR_101_Epoch_5_Batch_113720_Time_2019-08-08-07-49_checkpoint.pth', device))
    model_pool.append(init_model(IR_152, 'models/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth', device))
    model_pool.append(init_model(ResNet_50, 'models/Backbone_ResNet_50_Epoch_3_Batch_34116_Time_2019-08-02-19-12_checkpoint.pth', device))
    model_pool.append(init_model(ResNet_101, 'models/Backbone_ResNet_101_Epoch_4_Batch_90976_Time_2019-08-04-11-34_checkpoint.pth', device))
    return model_pool
 

# 产生一个对抗样本 
def main(origin_name, target_name, model_pool, device):
        
    crop_paths, gen_paths, imgs_dis, people = get_face.get_crop_face_file(get_face.crop_root, get_face.gen_root, gen=False)
    start_index = [0]
    s = 0
    for i in range(len(imgs_dis) - 1 ):
        s += imgs_dis[i]
        start_index.append(s)
    
    people_map = utils.list2map(people)
    origin_person = people_map[origin_name]
    target_person = people_map[target_name]
    
    s1 = start_index[origin_person]
    s2 = s1 + imgs_dis[origin_person]
    same_person_list = crop_paths[s1:s2]
    s1 = start_index[target_person]
    s2 = s1 + imgs_dis[target_person]
    target_face_list = crop_paths[s1:s2]
    s = len(same_person_list)
    index = config.origin_img_indexes[origin_name]
    # index = random.randint(0, s - 1)
    # index = 0
    img_origin = same_person_list.pop(index)
    print(img_origin)
    img_origin = to_torch_tensor(Image.open(img_origin)) 
    img_origin = img_origin.unsqueeze_(0).to(device)
    same_person_pool = get_img_pool(same_person_list, device)
    target_face_pool = get_img_pool(target_face_list, device)
    gaussian_blur = get_gaussian_blur(kernel_size=5, device=device)

    generator = get_noise(model_pool,img_origin, gaussian_blur, target_face_pool, same_person_pool, 1, 1, device)
    
    for i in range(250):
        noise = next(generator)    
    
    return gaussian_blur(noise), img_origin


# 单GPU运行，产生多个对抗样本
def one_card_run(people_list, crop_paths, device): 
    model_pool = get_model_pool(device) 
    for origin_name in people_list:
        for target_name in people:
            if origin_name == target_name:
                continue
            noise, img_origin = main(origin_name, target_name, model_pool, device)
            l2_val = utils.get_statistics(noise, img_origin)
            adv_sample = utils.get_adv_sample(noise, img_origin)
            if os.path.exists('out/') is False:
                os.mkdir('out/')
            adv_sample.save('out/%s_%s_%0.3f.png' % (origin_name, target_name, l2_val.item()))

# 多GPU运行行，产生多个对抗样本            
if __name__ == '__main__':
    # origin_name = sys.argv[1]
    crop_paths, gen_paths, imgs_dis, people = get_face.get_crop_face_file(get_face.crop_root, get_face.gen_root, gen=False)
    n = 3
    rate = len(people) // 3
    for i in range(n):
        if i == n - 1:
            people_list = people[(rate * i):]
        else:
            people_list = people[(rate * i): (rate * (i + 1))]
        device = torch.device('cuda:' + str(i))  
        process = multiprocessing.Process(target=one_card_run,args=(people_list, crop_paths, device))
        process.start()
    
    print(time.time() - t1)