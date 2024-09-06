#!/usr/bin/env/ python
# config: utf-8

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn
import torch.optim 
import NN_multi, NN_weight

# 定义数据集类
class GalaxyDataset(Dataset):

    def __init__(self, specimen_path, label_path): # 只测i波段的结果

        self.label = self.load_label(label_path) # 这次我不归一化，（取出来之后注意归一化哦）

        self.specimen_g, self.specimen_r, self.specimen_i = self.load_specimen(specimen_path)
        #self.specimen_i = self.load_specimen(specimen_path)

    def __len__(self):

        return self.label.shape[0] # 所有case的数量

    def __getitem__(self, idx):

        specimen_g = self.specimen_g[idx]

        specimen_r = self.specimen_r[idx]
        
        specimen_i = self.specimen_i[idx] 
        
        label = self.label[idx]
        
        return specimen_g, specimen_r, specimen_i, label 

    # 验证集的归一化因子需要使用训练集的最大值    
    def load_specimen(self, specimen_path):
        
        # 这里才是样本，注意它的输入参数数量

        specimens_g = np.zeros((self.__len__(), 2000, 5)) # 注意这里！
        
        specimens_r = np.zeros((self.__len__(), 2000, 5))

        specimens_i = np.zeros((self.__len__(), 2000, 5))
        
        for i in range(self.__len__()):
         
            data_g = np.loadtxt(f'{specimen_path}g/case_{i}.txt')
            
            data_r = np.loadtxt(f'{specimen_path}r/case_{i}.txt')

            data_i = np.loadtxt(f'{specimen_path}i/case_{i}.txt')

            specimens_g[i] = data_g
            
            specimens_r[i] = data_r

            specimens_i[i] = data_i

        specimens_g = torch.from_numpy(specimens_g) 

        specimens_r = torch.from_numpy(specimens_r)

        specimens_i = torch.from_numpy(specimens_i)

        specimens_g = specimens_g.to(dtype=torch.float32)

        specimens_r = specimens_r.to(dtype=torch.float32)

        specimens_i = specimens_i.to(dtype=torch.float32)

        return specimens_g, specimens_r, specimens_i

    def load_label(self, label_path):
    
        label = np.loadtxt(f'{label_path}sorted_shear.txt')
        
        label = label[:, :-1] # 注意，这是标签，不是输入参数
        
        label = torch.from_numpy(label) 
        
        label = label.to(dtype=torch.float32)
        
        label = label.unsqueeze(-1)

        return label

# 创建数据集对象和创建数据加载器

#指明训练集路径

# 训上几个模型

#t = 'train_disk_noCr_noNoise_noPsf'
t = 'train_disk_noCr_yesNoise_noPsf'

specimen_path_t = f'/Users/user/CRCNN_and_CRNN/NN_color_disk_cr/{t}/specimen/'

label_path_t = f'/Users/user/CRCNN_and_CRNN/NN_color_disk_cr/{t}/label/'

#指明验证集路径

#specimen_path_v = '/home/lqy/CRCNN_and_CRNN/NN_color_disk_cr/validation/specimen/'

#label_path_v = '/home/lqy/CRCNN_and_CRNN/NN_color_disk_cr/validation/label/'

# 声明运行设备，没GPU就使用CPU
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
#device = torch.device('cuda')
# 检查是否支持 MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 初始化ResNet模型和优化器
model_g2 = NN_multi.NN()
model_g2w = NN_weight.NN()

# 将模型转移到GPU
model_g2.to(device)
model_g2w.to(device)
#'''
# 加载模型，如果需要的话
#checkpoint_g2 = torch.load(f'/home/lqy/NN_disk_noCr_noNoise_noPsf_g2.pth') # 注意名称
checkpoint_g2 = torch.load(f'/Users/user/NN_disk_noCr_yesNoise_noPsf_g2.pth', map_location=torch.device('cpu')) # 注意名称
checkpoint_g2w = torch.load(f'/Users/user/NN_disk_noCr_noNoise_noPsf_g2w.pth', map_location=torch.device('cpu')) # 注意名称

#加载模型权重
model_g2.load_state_dict(checkpoint_g2['model_state_dict'])
model_g2w.load_state_dict(checkpoint_g2w['model_state_dict'])
#'''

# 定义优化器
optimizer = torch.optim.LBFGS(model_g2w.parameters(), line_search_fn='strong_wolfe') #注意改这里的模型

# 定义损失函数
criterion = torch.nn.MSELoss()

#创建训练数据集
dataset_t = GalaxyDataset(specimen_path_t, label_path_t)

#创建验证数据集
#dataset_v = GalaxyDataset(specimen_path_v, label_path_v)

# 设置批量大小
#batch_size_list = [15000, 1500, 15000, 1500, 15000, 1500, 15000, 1500, 15000, 1500, 15000, 1500, 15000]
batch_size_list = [7500, 3750, 7500, 3750, 7500, 3750, 7500, 3750, 7500, 3750, 7500, 3750, 7500]
#batch_size_list = [15000, 15000, 15000, 15000, 15000, 15000, 15000]
#print('1')
# 注意这里的起始索引
for j in range(0, len(batch_size_list)):
    
    #创建数据集加载器
    batch_size_t = batch_size_list[j]
    print(f'batch size: {batch_size_t}')

    dataloaders_t = DataLoader(dataset_t, batch_size=batch_size_t, shuffle=True, drop_last=True)
    #'''
    if j == 0:

        losses_g2_t = []

        start_epoch = 0
  
    elif j != 0:

        #加载模型
        #checkpoint_g2 = torch.load(f'/home/lqy/NN_disk_noCr_noNoise_noPsf_g2.pth') # 注意名称
        #checkpoint_g2w = torch.load(f'/home/lqy/NN_disk_noCr_noNoise_noPsf_g2w.pth') # 注意名称
        checkpoint_g2 = torch.load(f'/Users/user/NN_disk_noCr_yesNoise_noPsf_g2.pth') # 注意名称
        checkpoint_g2w = torch.load(f'/Users/user/NN_disk_noCr_yesNoise_noPsf_g2w.pth') # 注意名称
        
        #加载模型权重
        model_g2.load_state_dict(checkpoint_g2['model_state_dict'])
        model_g2w.load_state_dict(checkpoint_g2w['model_state_dict']) # 注意改这里

        #加载优化器状态
        #optimizer.load_state_dict(checkpoint_g2['optimizer_state_dict'])
        optimizer.load_state_dict(checkpoint_g2w['optimizer_state_dict'])

        #加载起始迭代次数
        #start_epoch = checkpoint_g2['epoch'] # 这个两者都一样
        start_epoch = checkpoint_g2w['epoch']

        #加载之前的损失值列表
        #losses_g2_t = np.load(f'/home/lqy/NN_disk_noCr_noNoise_noPsf_g2.npy')
        #losses_g2_t = np.load(f'/home/lqy/NN_disk_noCr_noNoise_noPsf_g2w.npy')
        #losses_g2_t = np.load(f'/home/lqy/NN_disk_yesCr_yesNoise_yesPsf_g2.npy')
        losses_g2_t = np.load(f'/Users/user/NN_disk_noCr_yesNoise_noPsf_g2w.npy')

        #losses_g1_v = np.load(f'/home/lqy/NN_disk_3fc_cr_g1_i_nonorm_v_{q_index}.npy') #注意名称
        #losses_g2_v = np.load(f'/home/lqy/NN_disk_3fc_cr_g2_i_nonorm_v_{q_index}.npy')

    model_g2.eval() # 注意更换
    model_g2w.train() # 注意更换

    if batch_size_t == 7500:
        end_epochs = start_epoch + 50
    else:
        end_epochs = start_epoch + 1

    # 这里是训练代码，包含测试过程
    for epoch in range(start_epoch, end_epochs):

        epoch_loss_g2_t = 0.0

        for batch_idx, (specimens_g, specimens_r, specimens_i, labels) in enumerate(dataloaders_t):
        #for batch_idx, (specimens_i, labels) in enumerate(dataloaders_t):    

            specimens_g = specimens_g.to(device)

            specimens_r = specimens_r.to(device)

            specimens_i = specimens_i.to(device)

            specimens = torch.cat((specimens_g, specimens_r, specimens_i), dim=-1)

            specimens_flattened = specimens.view(batch_size_t * 2000, 15)
            
            labels = labels.to(device)
            #print(specimens_i.shape)
            #specimens_g_flattened = specimens_g.view(batch_size_t * 4000, 5) # 注意
            #specimens_r_flattened = specimens_r.view(batch_size_t * 4000, 5)
            #specimens_i_flattened = specimens_i.view(batch_size_t * 2000, 5)

            def closure2():
                
                optimizer.zero_grad() # 注意更换
                #'''
                outputs_g2 = model_g2(specimens_flattened)
                outputs_g2w = model_g2w(specimens_flattened)

                outputs_reshaped_g2 = outputs_g2.view(batch_size_t, 2000, 1)
                outputs_reshaped_g2w = outputs_g2w.view(batch_size_t, 2000, 1)

                #g2_hat = torch.mean(outputs_reshaped_g2, dim=1)
                
                numerator = torch.bmm(outputs_reshaped_g2.transpose(-1, -2), outputs_reshaped_g2w).squeeze(-1)
                denominator = torch.sum(outputs_reshaped_g2w, dim=1)
                g2_hat = numerator/denominator
                #print(g2_hat)
                #print(labels[:, 1])
            
                loss_g2 = criterion(g2_hat, labels[:, 1])                
                #'''
                '''
                outputs_g2 = model_g2(specimens_flattened)

                outputs_reshaped_g2 = outputs_g2.view(batch_size_t, 2000, 1)

                g2_hat = torch.mean(outputs_reshaped_g2, dim=1)

                #print(g2_hat)
                #print(labels[:, 1])
            
                loss_g2 = criterion(g2_hat, labels[:, 1])                
                '''
                loss_g2.backward()

                return loss_g2
            
            loss_g2_t = optimizer.step(closure2) # 注意更换
        
            epoch_loss_g2_t += loss_g2_t.item()
        
        epoch_loss_g2_t = epoch_loss_g2_t/len(dataloaders_t)

        losses_g2_t = np.append(losses_g2_t, epoch_loss_g2_t)
        '''
        checkpoint_g2  = {
            'epoch': epoch + 1,
            'model_state_dict': model_g2.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict()
        } #注意保存的是哪个模型
        '''
        #'''
        checkpoint_g2w  = {
            'epoch': epoch + 1,
            'model_state_dict': model_g2w.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict()
        } #注意保存的是哪个模型
        #'''        
        #torch.save(checkpoint_g2, f'/home/lqy/NN_disk_yesCr_yesNoise_yesPsf_g2.pth')
        #torch.save(checkpoint_g2w, f'/home/lqy/NN_disk_noCr_noNoise_noPsf_g2w.pth')
        #torch.save(checkpoint_g2, f'/Users/user/NN_disk_noCr_yesNoise_noPsf_g2.pth')
        torch.save(checkpoint_g2w, f'/Users/user/NN_disk_noCr_yesNoise_noPsf_g2w.pth')

        #np.save(f'/home/lqy/NN_disk_noCr_noNoise_noPsf_g2.npy', losses_g2_t)
        #np.save(f'/home/lqy/NN_disk_noCr_noNoise_noPsf_g2w.npy', losses_g2_t)
        #np.save(f'/home/lqy/NN_disk_yesCr_yesNoise_yesPsf_g2.npy', losses_g2_t)
        #np.save(f'//Users/user/NN_disk_noCr_yesNoise_noPsf_g2.npy', losses_g2_t)
        np.save(f'//Users/user/NN_disk_noCr_yesNoise_noPsf_g2w.npy', losses_g2_t)

        print('train, epoch {}, g2 loss {}'.format(epoch, epoch_loss_g2_t))
        
        '''
        # 从这开始这批次训练结束
        
        # 从这开始验证
        model_g1.eval() # 注意更换
        
        model_g2.eval() # 注意更换

        epoch_loss_g1 = 0.0

        epoch_loss_g2 = 0.0

        #for batch_idx, (specimens_g, specimens_r, specimens_i, labels) in enumerate(dataloaders_v):
        for batch_idx, (specimens_i, labels) in enumerate(dataloaders_v):    

            #specimens_g = specimens_g.to(device)

            #specimens_r = specimens_r.to(device)

            specimens_i = specimens_i.to(device)

            #specimens_g = specimens_g.unsqueeze(2)

            #specimens_r = specimens_r.unsqueeze(2) 

            #specimens_i = specimens_i.unsqueeze(2)

            #specimens = torch.cat((specimens_g, specimens_r, specimens_i), dim=2)
            
            #specimens = torch.cat((specimens_g, specimens_r, specimens_i), dim=-1)
            #specimens_flattened = specimens.view(batch_size_t * 2000, 15)
            
            labels = labels.to(device)
            #print(specimens_i.shape)
            #specimens_g_flattened = specimens_g.view(batch_size_t * 4000, 5) # 注意
            #specimens_r_flattened = specimens_r.view(batch_size_t * 4000, 5)
            specimens_i_flattened = specimens_i.view(batch_size_t * 4000, 5)
            #specimens_g_flattened = specimens_g.view(batch_size_t * 100000, 5) # 注意
            #specimens_r_flattened = specimens_r.view(batch_size_t * 100000, 5)
            #specimens_i_flattened = specimens_i.view(batch_size_t * 100000, 5)        
                
            #outputs_g1 = model_g1(specimens_g_flattened, specimens_r_flattened, specimens_i_flattened) # 注意
            outputs_g1 = model_g1(specimens_i_flattened)

            outputs_reshaped_g1 = outputs_g1.view(batch_size_t, 4000, 1)
            #outputs_reshaped_g1 = outputs_g1.view(batch_size_t, 100000, 1)

            #loss_g1 = criterion1(torch.mean(torch.mean(outputs_reshaped_g1, dim=2), dim=1), labels[:, 0]) # 注意最后这个
            
            loss_g1 = criterion1(torch.mean(outputs_reshaped_g1, dim=1), labels[:, 0])


            #outputs_g2 = model_g2(specimens_g_flattened, specimens_r_flattened, specimens_i_flattened)
            outputs_g2 = model_g2(specimens_i_flattened)

            outputs_reshaped_g2 = outputs_g2.view(batch_size_t, 4000, 1)
            #outputs_reshaped_g2 = outputs_g2.view(batch_size_t, 100000, 1)

            #loss_g2 = criterion2(torch.mean(torch.mean(outputs_reshaped_g2, dim=2), dim=1), labels[:, 1]) # 注意最后这个
            
            loss_g2 = criterion2(torch.mean(outputs_reshaped_g2, dim=1), labels[:, 1])


            epoch_loss_g1 += loss_g1.item()
        
            epoch_loss_g2 += loss_g2.item()

        epoch_loss_g1 = epoch_loss_g1/len(dataloaders_t)
        
        epoch_loss_g2 = epoch_loss_g2/len(dataloaders_t)

        losses_g1_v = np.append(losses_g1_v, epoch_loss_g1)
        losses_g2_v = np.append(losses_g2_v, epoch_loss_g2)
        
        
        np.save(f'/home/lqy/NN_disk_3fc_cr_g1_i_nonorm_v.npy', losses_g1_v)

        
        np.save(f'/home/lqy/NN_disk_3fc_cr_g2_i_nonorm_v.npy', losses_g2_v)

        print('validation, epoch {}, g1 loss {}'.format(epoch, epoch_loss_g1))

        print('validation, epoch {}, g2 loss {}'.format(epoch, epoch_loss_g2))        
        '''

    print('\n')


    
