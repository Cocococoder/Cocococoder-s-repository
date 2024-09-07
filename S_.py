#!/usr/bin/env python
# config: utf-8

import numpy as np
import galsim, os
import multiprocessing
from multiprocessing import Manager
import random

def c(image, size):
    
    ctx = cty = size / 2
    
    Q0 = np.sum(image)
    
    rows, cols = np.indices((size, size))
    
    sumdli = np.sum(np.sum(image * (rows - ctx), axis=1))
    sumdlj = np.sum(np.sum(image * (cols - cty), axis=0))

    ctx += sumdli / Q0
    cty += sumdlj / Q0

    return ctx, cty, Q0

def r(image, size, ctx, cty, Q0):

    rows, cols = np.indices((size, size))

    Q11 = np.sum(image * (rows - ctx) ** 2)/Q0
    Q22 = np.sum(image * (cols - cty) ** 2)/Q0

    T = Q11 + Q22

    size = (T / 2) ** 0.5

    return size, T, Q11, Q22

def e(image, size, ctx, cty, T, Q11, Q22, Q0):

    rows, cols = np.indices((size, size))

    Q12 = np.sum(image * (rows - ctx) * (cols - cty))/Q0

    e1 = (Q11 - Q22) / T

    e2 = 2.0 * Q12 / T
        
    return e1, e2

def rho4(image, size, ctx, cty, Q0):

    # 生成图像的网格坐标
    x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))

    # 计算每个像素到图像中心的距离
    r = np.sqrt((x - ctx) ** 2 + (y - cty) ** 2)

    # 计算kurtosis
    kurtosis = np.sum((r / size) ** 4 * image) / Q0

    return kurtosis

def random_number():
    # 定义1到181的整数数组
    numbers = np.arange(1, 90)
    # 定义概率密度函数， 这里使用倒数函数f(x)=1/x，x从1到181
    pdf = 1.0 / numbers
    # 将概率密度函数标准化为总和为1
    pdf /= pdf.sum()
    # 使用numpy的choice函数从数组numbers中随机选择一个数，概率由pdf决定
    return np.random.choice(numbers, p=pdf)

def generate_random_output():
    # 生成包含1到100的整数的列表
    #num_list = list(range(1, 101))
    
    # 从列表中随机选择一个数
    rand_num = random.choice(num_list)
    
    # 检查随机数是否在?到?之间
    if rand_num >= 1 and rand_num <= 5:
        return 1
    else:
        return 0

#生成训练集

# 设置参数
pixel_scale = 0.074 # 注意这里！
max_e = 0.4 # 注意这里！!
max_g = 0.1
image_size = 32
band = np.array(['g','r','i'])
# 创建一个 GSParams 对象，并设置最大迭代次数
#my_params = galsim.hsm.HSMParams(max_mom2_iter=8000)
redshift = np.arange(0.0, 2.1, 0.1)
band_limit = np.array([[380, 580],[510, 720],[660, 900]])
# 生成包含1到100的整数的列表
num_list = list(range(1, 101))

#'''
# 把sed预先装进内存里
sed_data = {}
for i in band:
    sed_data[i] = {}
    for z in redshift:
        z = np.round(z, 1)
        
        El = np.loadtxt(f'./sed/galaxy/norm_sed/El_{i}_z{z}_n.txt')
        El = El[:, 1]
        # 从随机数组中筛选出非零值
        El = El[El > 0.]
        # 用均值试试
        El = np.mean(El)

        Im = np.loadtxt(f'./sed/galaxy/norm_sed/Im_{i}_z{z}_n.txt')
        Im = Im[:, 1]
        # 从随机数组中筛选出非零值
        Im = Im[Im > 0.]
        # 用均值试试
        Im = np.mean(Im)

        sed_data[i][z] = {'El': El, 'Im': Im}
#'''

# 从给定数组中随机选择images_number个数
#new_Ell_flux_list = np.random.choice(new_Ell_flux_list, size=images_number, replace=False)


# 可以通过不同的产生函数产生星系图像，故意看看它们参数范围有啥区别，如果区别不大就用最简单的跑，否则，个跑一部分

#这是训练集
#'''
t = 'train_disk_noCr_yesNoise_yesPsf'
#t = 'validation_augment_test'

for i in range(0, 3):
    for j in range(0, 3):
        #file_path = f"/Users/user/CRCNN_and_CRNN/NN_color_disk_cr/{t}/specimen/{j}/{band[i]}/"
        file_path = f"/home/lqy/CRCNN_and_CRNN/NN_color_disk_cr/{t}/specimen/{j}/{band[i]}/"
        # 检查上级目录是否存在，如果不存在则创建
        os.makedirs(file_path, exist_ok=True)

#file_path = f"/Users/user/CRCNN_and_CRNN/NN_color_disk_cr/{t}/label/"
file_path = f"/home/lqy/CRCNN_and_CRNN/NN_color_disk_cr/{t}/label/"
# 检查上级目录是否存在，如果不存在则创建
os.makedirs(file_path, exist_ok=True)



# 创建一个锁
#file_lock = threading.Lock()

#def process_first_galaxy(e, phi, b, z, bulge, disk, shear, case, output_queue, psf):
#def process_first_galaxy(e, phi, b, sed, disk, psf, shear, case, output_queue):
#def process_first_galaxy(e, phi, b, sed, disk, psf, shear, case, shared_data):
#def process_first_galaxy(e, phi, b, sed, disk, shear, case, shared_data):
#def process_first_galaxy(e_modulus, phi, b, sed, disk, snr, shear, case):
def process_first_galaxy(e_modulus, phi, b, sed, disk, snr, shear, psf, case):
    ##########################################
    # 设置固有椭率
    ellipticity_plus = galsim.Shear(e=e_modulus, beta=phi*galsim.degrees)

    #先制作第一个星系
    galaxy = sed * disk
    galaxy_plus = galaxy.shear(ellipticity_plus) # 添加固有椭率                                    
    galaxy_plus = galaxy_plus.shear(shear)
    galaxy_plus = galsim.Convolve([galaxy_plus, psf])

    '''
    psf = galsim.Airy(flux=1., lam=np.random.randint(band_limit[b][0], band_limit[b][1]), diam=2., obscuration=0.1)
    
    #final_epsf_image = psf.drawImage(scale=0.2)

    #和psf卷积
    galaxy_plus = galsim.Convolve([galaxy_plus, psf])
    
    picture_plus_no_noise = galaxy_plus.drawImage(image=galsim.ImageF(image_size, image_size, scale=pixel_scale))
    
    # 从这开始
    moments_plus_no_noise = galsim.hsm.FindAdaptiveMom(picture_plus_no_noise, hsmparams=my_params)
    
    sigma_ = moments_plus_no_noise.moments_sigma
    
    flux_ = moments_plus_no_noise.moments_amp
    
    shared_data[band[b]].append(sigma_)
    shared_data[band[b]].append(flux_)
    #print(shared_data[band[b]])
    '''
    # 随时删除
    picture_plus_no_noise = galaxy_plus.drawImage(image=galsim.ImageF(image_size, image_size, scale=pixel_scale))
    # 定义函数，用于生成带有噪声的样本并保存
    def generate_and_save_sample(case, t, b, flag5, q):

        try:

            #把图像的值读出来
            array = picture_plus_no_noise.array

            #'''
            average_intensity = np.mean(array)
            # 噪声的标准差
            noise_sigma = average_intensity/snr
            
            noise = noise_sigma * np.random.randn(image_size, image_size)
            
            array += noise #
            #'''
            '''
            if b == 2 and flag5 == 1:
                # 随机选择一个起始点的坐标
                start_x = np.random.randint(3*image_size/7, 4*image_size/7)
                start_y = np.random.randint(3*image_size/7, 4*image_size/7)
                #print(start_x, start_y)
                # 随机选择一个方向（0°到180°之间任意角度）
                angle = np.random.uniform(0, 180)

                # 将角度转换为弧度
                angle_rad = np.radians(angle)

                # 获取图像中最亮像素的值
                max_pixel_value = np.max(array) # 两个图一样，用哪个都行

                # 设置连续像素点的数量，n+1才是你想要的那个长度，比如当n=0，就对应1个像素！！！
                n = random_number() 

                # 计算直线终点的坐标
                end_x = start_x + n * np.cos(angle_rad)
                end_y = start_y + n * np.sin(angle_rad)

                # 在起始点和终点之间插值生成连续的像素点
                line_x = np.linspace(start_x, end_x, num=n+1)
                line_y = np.linspace(start_y, end_y, num=n+1)

                random_pixel_value = np.random.uniform(low=0.1*max_pixel_value, high=10*max_pixel_value)

                # 在图像上标记直线上的所有像素
                for i in range(len(line_x)):
                    x = int(round(line_x[i]))
                    y = int(round(line_y[i]))
                    if 0 <= x < image_size and 0 <= y < image_size:
                        # 产生最亮像素值的0.1-10倍之间的随机数
                        array[x, y] += random_pixel_value # 宇宙线往噪声上盖
                        #gal_minus_p[x, y] += random_pixel_value # 宇宙线往噪声上盖
                    else:
                        break

            '''
            # 我们需要的训练样本是包含噪声的，因此在添加完噪声后还要再测一次
            #moments_plus_yes_noise = galsim.hsm.FindAdaptiveMom(picture_plus, hsmparams=my_params)
            #picture_plus = galaxy_plus.drawImage(image=galsim.ImageF(image_size, image_size, scale=pixel_scale))
            #moments_plus_yes_noise = galsim.hsm.FindAdaptiveMom(picture_plus, hsmparams=my_params)
            '''
            moments_plus_yes_noise = galsim.Image(array).FindAdaptiveMom(hsmparams=my_params)

            e1, e2 = moments_plus_yes_noise.observed_shape.e1, moments_plus_yes_noise.observed_shape.e2
            
            sigma = moments_plus_yes_noise.moments_sigma

            flux = moments_plus_yes_noise.moments_amp

            rho4 = moments_plus_yes_noise.moments_rho4
            '''
            ctx, cty, Q0 = c(array, image_size)
            
            size, T, Q11, Q22 = r(array, image_size, ctx, cty, Q0)
            
            e1, e2 = e(array, image_size, ctx, cty, T, Q11, Q22, Q0)
            
            kurtosis = rho4(array, size, ctx, cty, Q0)

            arr = np.array([e1, e2, size, Q0, kurtosis])

            # 存specimen
            with open(f'/home/lqy/CRCNN_and_CRNN/NN_color_disk_cr/{t}/specimen/{q}/{band[b]}/case_{case}.txt', 'a') as file:
            #with open(f'/Users/user/CRCNN_and_CRNN/NN_color_disk_cr/{t}/specimen/{q}/{band[b]}/case_{case}.txt', 'a') as file:        
                np.savetxt(file, [arr])
        
            return True
        
        except Exception as error:
            #print(f"An error occurred in the band {band[b]}, sigma_ is: {shared_data[band[b]][0]}, snr is: {snr}")
            print(f"An error occurred in the band {band[b]}")
            print(error)
            '''
            if b == 2 and flag5 == 1:
                print(f"An error occurred, start coodrdinates:{start_x, start_y}, end coordinates:{end_x, end_y}, random_pixel_value:{random_pixel_value}")
            '''
            return False          

    for q in range(3):      
        
        flag5 = generate_random_output()
        
        while True:
            
            flag = generate_and_save_sample(case, t, b, flag5, q)

            if flag == True:
                break    
    
    ###########################################
    #再制作第二个星系，生成每个波长处真实的图像
    # 设置固有椭率
    ellipticity_minus = galsim.Shear(e=e_modulus, beta=(phi+90)*galsim.degrees)

    #先制作第一个星系
    galaxy = sed * disk
    galaxy_minus = galaxy.shear(ellipticity_minus) # 添加固有椭率                                    
    galaxy_minus = galaxy_minus.shear(shear)
    galaxy_minus = galsim.Convolve([galaxy_minus, psf])

    '''
    psf = galsim.Airy(flux=1., lam=np.random.randint(band_limit[b][0], band_limit[b][1]), diam=2., obscuration=0.1)

    #和psf卷积
    galaxy_minus = galsim.Convolve([galaxy_minus, psf])
      
    picture_minus_no_noise = galaxy_minus.drawImage(image=galsim.ImageF(image_size, image_size, scale=pixel_scale))
    
    # 从这开始
    #moments_minus_no_noise = galsim.hsm.FindAdaptiveMom(picture_minus_no_noise, hsmparams=my_params)
    
    #sigma = moments_minus_no_noise.moments_sigma
    
    #flux = moments_minus_no_noise.moments_amp
    '''
    # 随时删除
    picture_minus_no_noise = galaxy_minus.drawImage(image=galsim.ImageF(image_size, image_size, scale=pixel_scale))    
    def generate_and_save_sample(case, t, b, flag5, q):

        try:    
            #把图像的值读出来
            array = picture_minus_no_noise.array

            #'''
            average_intensity = np.mean(array)
            # 噪声的标准差
            noise_sigma = average_intensity/snr
            
            noise = noise_sigma * np.random.randn(image_size, image_size)
            
            array += noise #
            #'''
            '''
            if b == 2 and flag5 == 1:
                # 随机选择一个起始点的坐标
                start_x = np.random.randint(3*image_size/7, 4*image_size/7)
                start_y = np.random.randint(3*image_size/7, 4*image_size/7)
                #print(start_x, start_y)
                # 随机选择一个方向（0°到180°之间任意角度）
                angle = np.random.uniform(0, 180)

                # 将角度转换为弧度
                angle_rad = np.radians(angle)

                # 获取图像中最亮像素的值
                max_pixel_value = np.max(array) # 两个图一样，用哪个都行

                # 设置连续像素点的数量，n+1才是你想要的那个长度，比如当n=0，就对应1个像素！！！
                n = random_number() 

                # 计算直线终点的坐标
                end_x = start_x + n * np.cos(angle_rad)
                end_y = start_y + n * np.sin(angle_rad)

                # 在起始点和终点之间插值生成连续的像素点
                line_x = np.linspace(start_x, end_x, num=n+1)
                line_y = np.linspace(start_y, end_y, num=n+1)

                random_pixel_value = np.random.uniform(low=0.1*max_pixel_value, high=10*max_pixel_value)

                # 在图像上标记直线上的所有像素
                for i in range(len(line_x)):
                    x = int(round(line_x[i]))
                    y = int(round(line_y[i]))
                    if 0 <= x < image_size and 0 <= y < image_size:
                        # 产生最亮像素值的0.1-10倍之间的随机数
                        array[x, y] += random_pixel_value # 宇宙线往噪声上盖
                        #gal_minus_p[x, y] += random_pixel_value # 宇宙线往噪声上盖
                    else:
                        break
                        
            '''
            # 我们需要的训练样本是包含噪声的，因此在添加完噪声后还要再测一次
            #moments_plus_yes_noise = galsim.hsm.FindAdaptiveMom(picture_plus, hsmparams=my_params)
            #picture_plus = galaxy_plus.drawImage(image=galsim.ImageF(image_size, image_size, scale=pixel_scale))
            #moments_plus_yes_noise = galsim.hsm.FindAdaptiveMom(picture_plus, hsmparams=my_params)
            '''
            moments_minus_yes_noise = galsim.Image(array).FindAdaptiveMom(hsmparams=my_params)

            e1, e2 = moments_minus_yes_noise.observed_shape.e1, moments_minus_yes_noise.observed_shape.e2
            
            sigma = moments_minus_yes_noise.moments_sigma

            flux = moments_minus_yes_noise.moments_amp

            rho4 = moments_minus_yes_noise.moments_rho4
            '''
            ctx, cty, Q0 = c(array, image_size)
            
            size, T, Q11, Q22 = r(array, image_size, ctx, cty, Q0)
            
            e1, e2 = e(array, image_size, ctx, cty, T, Q11, Q22, Q0)
            
            kurtosis = rho4(array, size, ctx, cty, Q0)

            arr = np.array([e1, e2, size, Q0, kurtosis])                            

            # 存specimen
            with open(f'/home/lqy/CRCNN_and_CRNN/NN_color_disk_cr/{t}/specimen/{q}/{band[b]}/case_{case}.txt', 'a') as file:
            #with open(f'/Users/user/CRCNN_and_CRNN/NN_color_disk_cr/{t}/specimen/{q}/{band[b]}/case_{case}.txt', 'a') as file:        
                np.savetxt(file, [arr])
            
            return True
        
        except Exception as error:
            #print(f"An error occurred in the band {band[b]}, sigma_ is: {shared_data[band[b]][0]}, snr is: {snr}")
            print(f"An error occurred in the band {band[b]}")
            print(error)
            '''
            if b == 2 and flag5 == 1:
                print(f"An error occurred, start coodrdinates:{start_x, start_y}, end coordinates:{end_x, end_y}, random_pixel_value:{random_pixel_value}")
            '''
            return False
                        
    for q in range(3):      
        
        flag5 = generate_random_output()
        
        while True:
            
            flag = generate_and_save_sample(case, t, b, flag5, q)

            if flag == True:

                break      
        
    #output_queue.put([sigma_, flux_]) 
    #shared_data[band[b]] = [sigma_, flux_]

#def process_remaining_galaxy(e, phi, b, z, average_intensity, bulge, disk, shear, case, psf):
#def process_remaining_galaxy(e, phi, b, sed, disk, psf, shear, case, shared_data):
#def process_remaining_galaxy(e, phi, b, sed, disk, shear, case, shared_data):
#def process_remaining_galaxy(e_modulus, phi, b, sed, disk, snr, shear, case):
def process_remaining_galaxy(e_modulus, phi, b, sed, disk, snr, shear, psf, case):
    ##########################################
    # 设置固有椭率
    ellipticity_plus = galsim.Shear(e=e_modulus, beta=phi*galsim.degrees)

    #先制作第一个星系
    galaxy = sed * disk
    galaxy_plus = galaxy.shear(ellipticity_plus) # 添加固有椭率                                    
    galaxy_plus = galaxy_plus.shear(shear)
    galaxy_plus = galsim.Convolve([galaxy_plus, psf])
    '''
    psf = galsim.Airy(flux=1., lam=np.random.randint(band_limit[b][0], band_limit[b][1]), diam=2., obscuration=0.1)

    #和psf卷积
    galaxy_plus = galsim.Convolve([galaxy_plus, psf])
    
    picture_plus_no_noise = galaxy_plus.drawImage(image=galsim.ImageF(image_size, image_size, scale=pixel_scale))
    
    # 从这开始
    moments_plus_no_noise = galsim.hsm.FindAdaptiveMom(picture_plus_no_noise, hsmparams=my_params)
    
    sigma_ = moments_plus_no_noise.moments_sigma
    
    flux_ = moments_plus_no_noise.moments_amp
    
    shared_data[band[b]].append(sigma_)
    shared_data[band[b]].append(flux_)
    #print(shared_data[band[b]])
    '''
    # 随时删除
    picture_plus_no_noise = galaxy_plus.drawImage(image=galsim.ImageF(image_size, image_size, scale=pixel_scale))
    # 定义函数，用于生成带有噪声的样本并保存
    def generate_and_save_sample(case, t, b, flag5, q):

        try:

            #把图像的值读出来
            array = picture_plus_no_noise.array

            #'''
            average_intensity = np.mean(array)
            # 噪声的标准差
            noise_sigma = average_intensity/snr
            
            noise = noise_sigma * np.random.randn(image_size, image_size)
            
            array += noise #
            #'''
            '''
            if b == 2 and flag5 == 1:
                # 随机选择一个起始点的坐标
                start_x = np.random.randint(3*image_size/7, 4*image_size/7)
                start_y = np.random.randint(3*image_size/7, 4*image_size/7)
                #print(start_x, start_y)
                # 随机选择一个方向（0°到180°之间任意角度）
                angle = np.random.uniform(0, 180)

                # 将角度转换为弧度
                angle_rad = np.radians(angle)

                # 获取图像中最亮像素的值
                max_pixel_value = np.max(array) # 两个图一样，用哪个都行

                # 设置连续像素点的数量，n+1才是你想要的那个长度，比如当n=0，就对应1个像素！！！
                n = random_number() 

                # 计算直线终点的坐标
                end_x = start_x + n * np.cos(angle_rad)
                end_y = start_y + n * np.sin(angle_rad)

                # 在起始点和终点之间插值生成连续的像素点
                line_x = np.linspace(start_x, end_x, num=n+1)
                line_y = np.linspace(start_y, end_y, num=n+1)

                random_pixel_value = np.random.uniform(low=0.1*max_pixel_value, high=10*max_pixel_value)

                # 在图像上标记直线上的所有像素
                for i in range(len(line_x)):
                    x = int(round(line_x[i]))
                    y = int(round(line_y[i]))
                    if 0 <= x < image_size and 0 <= y < image_size:
                        # 产生最亮像素值的0.1-10倍之间的随机数
                        array[x, y] += random_pixel_value # 宇宙线往噪声上盖
                        #gal_minus_p[x, y] += random_pixel_value # 宇宙线往噪声上盖
                    else:
                        break
                        
            '''
            # 我们需要的训练样本是包含噪声的，因此在添加完噪声后还要再测一次
            #moments_plus_yes_noise = galsim.hsm.FindAdaptiveMom(picture_plus, hsmparams=my_params)
            #picture_plus = galaxy_plus.drawImage(image=galsim.ImageF(image_size, image_size, scale=pixel_scale))
            #moments_plus_yes_noise = galsim.hsm.FindAdaptiveMom(picture_plus, hsmparams=my_params)
            '''
            moments_plus_yes_noise = galsim.Image(array).FindAdaptiveMom(hsmparams=my_params)

            e1, e2 = moments_plus_yes_noise.observed_shape.e1, moments_plus_yes_noise.observed_shape.e2
            
            sigma = moments_plus_yes_noise.moments_sigma

            flux = moments_plus_yes_noise.moments_amp

            rho4 = moments_plus_yes_noise.moments_rho4
            '''
            ctx, cty, Q0 = c(array, image_size)
            
            size, T, Q11, Q22 = r(array, image_size, ctx, cty, Q0)
            
            e1, e2 = e(array, image_size, ctx, cty, T, Q11, Q22, Q0)
            
            kurtosis = rho4(array, size, ctx, cty, Q0)

            arr = np.array([e1, e2, size, Q0, kurtosis])                            

            # 存specimen
            with open(f'/home/lqy/CRCNN_and_CRNN/NN_color_disk_cr/{t}/specimen/{q}/{band[b]}/case_{case}.txt', 'a') as file:
            #with open(f'/Users/user/CRCNN_and_CRNN/NN_color_disk_cr/{t}/specimen/{q}/{band[b]}/case_{case}.txt', 'a') as file:        
                np.savetxt(file, [arr])
        
            return True
        
        except Exception as error:
            #print(f"An error occurred in the band {band[b]}, sigma_ is: {shared_data[band[b]][0]}, snr is: {snr}")
            print(f"An error occurred in the band {band[b]}")
            print(error)
            '''
            if b == 2 and flag5 == 1:
                print(f"An error occurred, start coodrdinates:{start_x, start_y}, end coordinates:{end_x, end_y}, random_pixel_value:{random_pixel_value}")
            '''
            return False   
                 
    for q in range(3):      
        
        flag5 = generate_random_output()
        
        while True:
            
            flag = generate_and_save_sample(case, t, b, flag5, q)

            if flag == True:
                break      
    
    ###########################################
    #再制作第二个星系，生成每个波长处真实的图像
    # 设置固有椭率
    ellipticity_minus = galsim.Shear(e=e_modulus, beta=(phi+90)*galsim.degrees)

    #先制作第一个星系
    galaxy = sed * disk
    galaxy_minus = galaxy.shear(ellipticity_minus) # 添加固有椭率                                    
    galaxy_minus = galaxy_minus.shear(shear)
    galaxy_minus = galsim.Convolve([galaxy_minus, psf])
    '''
    psf = galsim.Airy(flux=1., lam=np.random.randint(band_limit[b][0], band_limit[b][1]), diam=2., obscuration=0.1)

    #和psf卷积
    galaxy_minus = galsim.Convolve([galaxy_minus, psf])
      
    picture_minus_no_noise = galaxy_minus.drawImage(image=galsim.ImageF(image_size, image_size, scale=pixel_scale))
    
    # 从这开始
    #moments_minus_no_noise = galsim.hsm.FindAdaptiveMom(picture_minus_no_noise, hsmparams=my_params)
    
    #sigma = moments_minus_no_noise.moments_sigma
    
    #flux = moments_minus_no_noise.moments_amp
    '''
    # 随时删除
    picture_minus_no_noise = galaxy_minus.drawImage(image=galsim.ImageF(image_size, image_size, scale=pixel_scale))    
    def generate_and_save_sample(case, t, b, flag5, q):

        try:    
            
            #把图像的值读出来
            array = picture_minus_no_noise.array

            #'''
            average_intensity = np.mean(array)
            # 噪声的标准差
            noise_sigma = average_intensity/snr
            
            noise = noise_sigma * np.random.randn(image_size, image_size)
            
            array += noise #
            #'''
            '''
            if b == 2 and flag5 == 1:
                # 随机选择一个起始点的坐标
                start_x = np.random.randint(3*image_size/7, 4*image_size/7)
                start_y = np.random.randint(3*image_size/7, 4*image_size/7)
                #print(start_x, start_y)
                # 随机选择一个方向（0°到180°之间任意角度）
                angle = np.random.uniform(0, 180)

                # 将角度转换为弧度
                angle_rad = np.radians(angle)

                # 获取图像中最亮像素的值
                max_pixel_value = np.max(array) # 两个图一样，用哪个都行

                # 设置连续像素点的数量，n+1才是你想要的那个长度，比如当n=0，就对应1个像素！！！
                n = random_number() 

                # 计算直线终点的坐标
                end_x = start_x + n * np.cos(angle_rad)
                end_y = start_y + n * np.sin(angle_rad)

                # 在起始点和终点之间插值生成连续的像素点
                line_x = np.linspace(start_x, end_x, num=n+1)
                line_y = np.linspace(start_y, end_y, num=n+1)

                random_pixel_value = np.random.uniform(low=0.1*max_pixel_value, high=10*max_pixel_value)

                # 在图像上标记直线上的所有像素
                for i in range(len(line_x)):
                    x = int(round(line_x[i]))
                    y = int(round(line_y[i]))
                    if 0 <= x < image_size and 0 <= y < image_size:
                        # 产生最亮像素值的0.1-10倍之间的随机数
                        array[x, y] += random_pixel_value # 宇宙线往噪声上盖
                        #gal_minus_p[x, y] += random_pixel_value # 宇宙线往噪声上盖
                    else:
                        break
                    
            '''
            # 我们需要的训练样本是包含噪声的，因此在添加完噪声后还要再测一次
            #moments_plus_yes_noise = galsim.hsm.FindAdaptiveMom(picture_plus, hsmparams=my_params)
            #picture_plus = galaxy_plus.drawImage(image=galsim.ImageF(image_size, image_size, scale=pixel_scale))
            #moments_plus_yes_noise = galsim.hsm.FindAdaptiveMom(picture_plus, hsmparams=my_params)
            '''
            moments_minus_yes_noise = galsim.Image(array).FindAdaptiveMom(hsmparams=my_params)

            e1, e2 = moments_minus_yes_noise.observed_shape.e1, moments_minus_yes_noise.observed_shape.e2
            
            sigma = moments_minus_yes_noise.moments_sigma

            flux = moments_minus_yes_noise.moments_amp

            rho4 = moments_minus_yes_noise.moments_rho4
            '''
            ctx, cty, Q0 = c(array, image_size)
            
            size, T, Q11, Q22 = r(array, image_size, ctx, cty, Q0)
            
            e1, e2 = e(array, image_size, ctx, cty, T, Q11, Q22, Q0)
            
            kurtosis = rho4(array, size, ctx, cty, Q0)

            arr = np.array([e1, e2, size, Q0, kurtosis])                                        

            # 存specimen
            with open(f'/home/lqy/CRCNN_and_CRNN/NN_color_disk_cr/{t}/specimen/{q}/{band[b]}/case_{case}.txt', 'a') as file:
            #with open(f'/Users/user/CRCNN_and_CRNN/NN_color_disk_cr/{t}/specimen/{q}/{band[b]}/case_{case}.txt', 'a') as file:        
                np.savetxt(file, [arr])
            
            return True
        
        except Exception as error:
            #print(f"An error occurred in the band {band[b]}, sigma_ is: {shared_data[band[b]][0]}, snr is: {snr}")
            print(f"An error occurred in the band {band[b]}")
            print(error)
            '''
            if b == 2 and flag5 == 1:
                print(f"An error occurred, start coodrdinates:{start_x, start_y}, end coordinates:{end_x, end_y}, random_pixel_value:{random_pixel_value}")
            '''
            return False    

    for q in range(3):      
        
        flag5 = generate_random_output()
        
        while True:
            
            flag = generate_and_save_sample(case, t, b, flag5, q)

            if flag == True:

                break     


n_list = np.array([1.5, 1.2, 1.0])

#'''
def gen(case):
    
    # 重新设置随机数生成器的种子,这一点非常重要
    np.random.seed()  

    # 产生这个case所使用的星系的基本参数，每个case只用一个星系

    # 随机生成星系的椭率
    e1 = np.random.uniform(low=-max_e, high=+max_e)
    e2 = np.random.uniform(low=-max_e, high=+max_e)    
    e_modulus = (e1**2 + e2**2)**0.5

    # 随机生成星系的剪切，每个case只用一个shear
    g1 = np.random.uniform(low=-max_g, high=+max_g)
    g2 = np.random.uniform(low=-max_g, high=+max_g)    
    shear = galsim.Shear(e1=g1, e2=g2) # 现在要记这个g1和g2

    # 存label，每个case的shear一样
    # 将数组保存到文件
    arr = np.array([g1, g2, case])
    
    #with open(f'/Users/user/CRCNN_and_CRNN/NN_color_disk_cr/{t}/label/shear.txt', 'a') as file:
    with open(f'/home/lqy/CRCNN_and_CRNN/NN_color_disk_cr/{t}/label/shear.txt', 'a') as file:
    
        np.savetxt(file, [arr])

    # 如果能让label对星系的限制的简并变的越少越好，但这也会导致输入特征的增加 
    # 随机生成星系的尺寸
    disk_hlr = np.random.uniform(low=0.1, high=1.2)
    #bulge_hlr = np.random.uniform(low=0.09, high=0.2)

    disk_list = np.array([])

    # 选择核和盘
    for i in range(3):
        
        disk_list = np.append(disk_list, galsim.Sersic(flux=1., n=n_list[i], half_light_radius=disk_hlr))

    #disk = galsim.Sersic(flux=1., n=1.0, half_light_radius=disk_hlr)    
    #bulge = galsim.Sersic(flux=1., n=1.5, half_light_radius=bulge_hlr)    

    #with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    #with concurrent.futures.ProcessPoolExecutor() as executor:  
    #with multiprocessing.Pool() as pool:      

    # 现在把每个realization的psf都变得不一样
    # 为了使问题简化，在不给模型输入psf信息的情况下，每个case中每个realization的psf使用一样的 
    counts = 0 # 注意这里！
    #'''
    wavelength_g = np.random.randint(380, 580)
    wavelength_r = np.random.randint(510, 720)
    wavelength_i = np.random.randint(660, 900)
    #print(f'g波段psf所处的波长: {wavelength_g}, r波段psf所处的波长: {wavelength_r}, i波段psf所处的波长: {wavelength_i}')
    psf_g = galsim.Airy(flux=1., lam=wavelength_g, diam=2., obscuration=0.1)
    psf_r = galsim.Airy(flux=1., lam=wavelength_r, diam=2., obscuration=0.1)
    psf_i = galsim.Airy(flux=1., lam=wavelength_i, diam=2., obscuration=0.1)
    
    psf_list = np.array([psf_g, psf_r, psf_i])
    #'''
    # 选择星系SED的红移
    z = np.random.choice(redshift)
    z = np.round(z, 1)

    El_sed = [sed_data[band[b]][z]['El'] for b in range(3)]
    Im_sed = [sed_data[band[b]][z]['Im'] for b in range(3)]
    
    sed = El_sed + Im_sed

    #pool = multiprocessing.Pool(processes=3)
    # 使用 with 语句创建进程池对象
    #with Manager() as manager:

    #shared_data = manager.dict()
    #shared_data['g'] = manager.list()
    #shared_data['r'] = manager.list()
    #shared_data['i'] = manager.list()    

    # 第一对星系单独拿出来只不过是为了测一下平均强度
    while counts == 0:
        
        # 每一对星系的旋转角度得不一样
        phi = np.random.uniform(low=0, high=180)
        '''
        psf_list = [galsim.Airy(flux=1., lam=np.random.randint(band_limit[0][0], band_limit[0][1]), diam=2., obscuration=0.1), \
                    galsim.Airy(flux=1., lam=np.random.randint(band_limit[1][0], band_limit[1][1]), diam=2., obscuration=0.1), \
                    galsim.Airy(flux=1., lam=np.random.randint(band_limit[2][0], band_limit[2][1]), diam=2., obscuration=0.1)]
        '''
        # 三个波段信噪比得一样
        snr_list = np.array([])

        snr_certain_count = np.random.uniform(10,100)

        for i in range(3):

            snr_list = np.append(snr_list, snr_certain_count)

        # 为了抵消固有椭率，产生两个椭率的模一样，但是方位角相差90°的星系
        
        # 保证每个星系（realization）在三个波段的旋转角度相同

        #output_queue1 = Queue()
        #output_queue2 = Queue()
        #output_queue3 = Queue()
        
        #output_queue = np.array([output_queue1, output_queue2, output_queue3])
        
        #input_data_list = [(e, phi, j, sed[j], disk_list[j], psf_list[j], shear, case, [output_queue1, output_queue2, output_queue3][j]) for j in range(3)]
        #input_data_list = [(e, phi, j, sed[j], disk_list[j], psf_list[j], shear, case, shared_data) for j in range(3)]
        #input_data_list = [(e, phi, j, sed[j], disk_list[j], shear, case, shared_data) for j in range(3)]
        input_data_list = [(e_modulus, phi, j, sed[j], disk_list[j], snr_list[j], shear, psf_list[j], case) for j in range(3)]
        #input_data_list = [(e_modulus, phi, j, sed[j], disk_list[j], shear, psf_list[j], case) for j in range(3)]
        pool.starmap(process_first_galaxy, input_data_list)

        ############################################################
        
        counts += 1
        
        print(f'第{case}个case的第{counts}对星系产生完毕')
        
        '''
        queue1_data = output_queue1.get()
        queue2_data = output_queue2.get()
        queue3_data = output_queue3.get()
        output_queue1.close()
        output_queue2.close()
        output_queue3.close()
        sigma_list = [queue1_data[0], queue2_data[0], queue3_data[0]]
        flux_list = [queue1_data[1], queue2_data[1], queue3_data[1]]
        '''
        
        #sigma_list = [shared_data['g'][0], shared_data['r'][0], shared_data['i'][0]]
        #flux_list = [shared_data['g'][1], shared_data['r'][1], shared_data['i'][1]]

    #print('1')
    
    while counts < 1000: # 每个case创建1000对星系图像，注意这里，经常调节！
        
        # 每一对星系的旋转角度得不一样
        phi = np.random.uniform(low=0, high=180)
        '''
        psf_list = [galsim.Airy(flux=1., lam=np.random.randint(band_limit[0][0], band_limit[0][1]), diam=2., obscuration=0.1), \
                    galsim.Airy(flux=1., lam=np.random.randint(band_limit[1][0], band_limit[1][1]), diam=2., obscuration=0.1), \
                    galsim.Airy(flux=1., lam=np.random.randint(band_limit[2][0], band_limit[2][1]), diam=2., obscuration=0.1)]
        '''
        snr_list = np.array([])

        snr_certain_count = np.random.uniform(10,100)

        for i in range(3):

            snr_list = np.append(snr_list, snr_certain_count)

        #input_data_list = [(e, phi, j, sed[j], disk_list[j], psf_list[j], shear, case, shared_data) for j in range(3)]
        #input_data_list = [(e, phi, j, sed[j], disk_list[j], shear, case, shared_data) for j in range(3)]
        input_data_list = [(e_modulus, phi, j, sed[j], disk_list[j], snr_list[j], shear, psf_list[j], case) for j in range(3)]
        #input_data_list = [(e_modulus, phi, j, sed[j], disk_list[j], shear, psf_list[j], case) for j in range(3)]
        pool.starmap(process_remaining_galaxy, input_data_list)

        ##########################################

        counts += 1
        
        print(f'第{case}个case的第{counts}对星系产生完毕')  

    #pool.close()
    #pool.join()  


if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        # 任务分配的代码
        for i in range(0, 5000):

            gen(i)  
