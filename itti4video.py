import cv2
import argparse
import tqdm
import numpy as np
import os
from itti_saliency_python import *


def Itti_Saliency_map4img(image):
       
    resized_image = resize_to_normal_shape(image)
    # gaussian_img_list = eight_pyrimid_built(resized_image)

    b, g, r = seperate_RGB_chanells(resized_image)

    r_sigma = eight_pyrimid_built(r)
    g_sigma = eight_pyrimid_built(g)
    b_sigma = eight_pyrimid_built(b)
    I = [(r_sigma[i]+g_sigma[i]+b_sigma[i])/3 for i in range(9)]

    # I2 = (np.int16(b) + np.int16(g) + np.int16(r))/3
    # I2s = eight_pyrimid_built(I2)

    maximum = [np.max(I[i]) for i in range(9)]


    b = [np.where(b_sigma[i]>= 0.1 * maximum[i],b_sigma[i],0) for i in range(9)]
    g = [np.where(g_sigma[i]>= 0.1 * maximum[i],g_sigma[i],0) for i in range(9)]
    r = [np.where(r_sigma[i]>= 0.1 * maximum[i],r_sigma[i],0) for i in range(9)]

    Is = I
    Rs = [r[i]-(g[i]+b[i])/2 for i in range(9)]
    Gs = [g[i]-(r[i]+b[i])/2 for i in range(9)]
    Bs = [b[i]-(g[i]+r[i])/2 for i in range(9)]
    Ys = [(r[i]+g[i])/2 - np.abs(r[i] - g[i])/2 - b[i] for i in range(9)]
    
    Rs = [np.where(img>0,img,0) for img in Rs]
    Gs = [np.where(img>0,img,0) for img in Gs]
    Bs = [np.where(img>0,img,0) for img in Bs]
    Ys = [np.where(img>0,img,0) for img in Ys]
    kernel_0, kernel_45, kernel_90, kernel_135 = processing_gabor_filters()

    Os = [gabor_filter(Is[i],kernel_0,kernel_45,kernel_90,kernel_135) for i in range(9)]

    c_set = (2,3,4)
    delta_set = (3,4)
    theta_set = (0,45,90,135)

    I_dict = {}
    RG_dict = {}
    BY_dict = {}
    O_dict = {}
    for c in c_set:
        for delta in delta_set:
            I_dict[(c,c+delta)] = Is_scale(Is,c,c+delta)
            RG_dict[(c,c+delta)] = RG_scale(Rs,Gs,c,c+delta)
            BY_dict[(c,c+delta)] = BY_scale(Bs,Ys,c,c+delta)
            for theta in theta_set:
                O_dict[(c,c+delta,theta)] = O_c_s_theta(Os,c,c+delta,theta)
    I_bar = np.zeros((1,1))
    C_bar = np.zeros((1,1))
    O_bar_0 = np.zeros((1,1))
    O_bar_45 = np.zeros((1,1))
    O_bar_90 = np.zeros((1,1))
    O_bar_135 = np.zeros((1,1))

    addition_shape = (Is[4].shape[1],Is[4].shape[0])

    for c in c_set:
        for delta in delta_set:
            I_bar = addition(I_bar,normalize_img(I_dict[(c,c+delta)]),addition_shape)
            C_bar = addition(C_bar,normalize_img(RG_dict[(c,c+delta)]),addition_shape)
            C_bar = addition(C_bar,normalize_img(BY_dict[(c,c+delta)]),addition_shape)

            O_bar_0 = addition(O_bar_0,normalize_img(O_dict[(c,c+delta,0)]),addition_shape)
            O_bar_45 = addition(O_bar_45,normalize_img(O_dict[(c,c+delta,45)]),addition_shape)
            O_bar_90 = addition(O_bar_90,normalize_img(O_dict[(c,c+delta,90)]),addition_shape)
            O_bar_135 = addition(O_bar_135,normalize_img(O_dict[(c,c+delta,135)]),addition_shape)

    O_bar = np.zeros((1,1))
    for O_bar_theta in [O_bar_0, O_bar_45, O_bar_90, O_bar_135]:
        O_bar = addition(O_bar,normalize_img(O_bar_theta),addition_shape)
    S = (normalize_img(I_bar) + normalize_img(C_bar) + normalize_img(O_bar))/3

    return S


def parse_args():
    """
    get args.
    """
    parse = argparse.ArgumentParser(description='essential parameters') 
    # 
    parse.add_argument('--video_path', default="E:\\Li Lab\\itti_and_lif\\video\\video\\video\\230.avi", type=str, help='path of sample video') 
    parse.add_argument('--output', default="save_img", type=str, help='output type, save_img or test') 
    parse.add_argument('--generate_name', default="", type=str, help='path of generated video')
    
    args = parse.parse_args() 
    return args

def main(args:argparse.Namespace):
    """
    the main function of dynamic scene saliency map prediction.
    """
    folder_name = args.generate_name
    _name = folder_name.split("\\")[-1].split("/")[-1].split(".")[0]
    if os.path.exists(f"{folder_name}") == False:
        print(f'generate for {_name}')
        os.mkdir(f"{folder_name}")
    else:
        print(f"folder {_name} already exists!")
        quit()
    #------------------------------------------------initialize---------------------------------------------------------
    
    video_path = args.video_path
    video_cap = cv2.VideoCapture(video_path)
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_number = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    final_shape = (width,height)
    ret, first_frame = video_cap.read()
    if not ret:
        # check if loaded correctly
        raise RuntimeError(f"failed to load video! Check video path = {video_path}")
    #------------------------------------------------FirstImageProcessing---------------------------------------------------------
    # first we calculate the full gaussian pyramid.
    resized_image = resize_to_normal_shape(first_frame)
    # resized_image = cv2.resize(first_frame,(320,240),interpolation=cv2.INTER_NEAREST)
    static_saliency_map = Itti_Saliency_map4img(resized_image)
    # static_saliency_map = resize_to_normal_shape(static_saliency_map)
    static_saliency_map = cv2.resize(static_saliency_map,(224,224),interpolation=cv2.INTER_NEAREST)
    static_saliency_map = cv2.normalize(static_saliency_map, None, 0, 255, cv2.NORM_MINMAX) # minmax coding
    rgb = cv2.cvtColor(np.uint8(static_saliency_map),cv2.COLOR_GRAY2BGR)
    if args.output == 'save_img':
        cv2.imwrite(f"{folder_name}/{1:04d}.jpg",rgb)
    else:
        raise NotImplementedError
    #------------------------------------------------VideoProcessing---------------------------------------------------------

    for _count in tqdm.trange(int(frame_number)): #in tqdm.trange(int(frame_number)):
        ret, frame = video_cap.read()
        if not ret: # whole video is processed
            print("Done!")
            break
        S_bar = Itti_Saliency_map4img(frame)
        S_bar = cv2.resize(S_bar,(224,224),interpolation=cv2.INTER_NEAREST)
        S_bar = cv2.normalize(S_bar, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(np.uint8(S_bar),cv2.COLOR_GRAY2BGR)
        if args.output == 'save_img':
            cv2.imwrite(f"{folder_name}/{_count+2:04d}.jpg",rgb)
        else:
            raise NotImplementedError

    video_cap.release()
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    args = parse_args()
    main(args)
