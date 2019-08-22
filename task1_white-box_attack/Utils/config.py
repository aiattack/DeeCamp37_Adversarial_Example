
# 用指定的图片生成对抗样本
origin_img_indexes = {'chengqiang': 31,
                'wangqf':6,
                 'djw':113,
                 'xujc':7,
                 'zhulq':30,
                 'jianyc':16,
                 'zhangwt':26,
                 'xingyc':16,
                 'wt':63,
                 'fusl':17,
                 'lxj':72,
                 'guoth':14,
                 'zyy':38,
                 'qiuhy':51,
                 'jinlk':0,
                  'byx':13,
                 'liuyl':23,
                 'za':52,
                 'wat':17}


'''
这套参数可以比较高的准确的生成对抗样本，
但是容易生成与目标相似的人脸
其结果像人脸融合
噪声l2范数约6~7

注：loss1 为cos_los, loss2 为l2_normal, loss3 为二阶tv_loss， loss2, loss3 加在 gauss_blur(noise)上.
    clamp:[-25.5, 25.5],
'''
iter_hyperparameter1 = {loss_weight:[1, 0.0025, 0.004],
                     lr: 2}


iter_hyperparameter = iter_hyperparameter1

