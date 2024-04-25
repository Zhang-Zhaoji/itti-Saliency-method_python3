import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter

##-----------------------##
"""
If you want to use this code, you can just \n
from Itti_method import Itti_Saliency_map\n
then you can just use it as :\n
itti_saliency_map = Itti_Saliency_map("your_image_path", ifshow = False)\n
the parameter ifshow controls that if the 6 + 12 + 24 different maps are displayed.\n
Hope you enjoy it!\n

Caution: Sometimes CV2 may not be able to read your image, please change the image format or check if your image path is correct.
"""
##-----------------------##

def gaussian_pyrimid(image):
   """
   return /2 resolution gaussian pyrimid
   """
   return cv2.pyrDown(image)

def eight_pyrimid_built(image):
   """
   yielding horizontal and vertical image-reduction factors ranging from 1:1 (scale zero) to 1:256 (scale eight) in eight octaves.
   """
   image_list = []
   image2 = image.copy()
   for i in range(9):
      image_list.append(image2)
      image2 = gaussian_pyrimid(image2)
   return image_list

def resize_to_normal_shape(image):
    """
    Input is provided in the form of static color images, usually digitized at 640*480 resolution
    """
    resized_image = cv2.resize(image,(640,480),interpolation=cv2.INTER_LINEAR)
    return resized_image

def seperate_RGB_chanells(image):
   """
    Seperate RGB chanells to going through following algorithms
    """
   return image[:,:,0], image[:,:,1], image[:,:,2]


def subtraction(img1,img2,ifshow = False):
    """
    return img1 - img2 but resized to the higher resolution
    """
    if ifshow:
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
    shape = img1.shape if img1.shape[0] >= img2.shape[0] else img2.shape
    shape = (shape[1],shape[0])
    image_1 = cv2.resize(img1,shape,interpolation=cv2.INTER_LINEAR)
    image_2 = cv2.resize(img2,shape,interpolation=cv2.INTER_LINEAR)
    if ifshow:
        plt.imshow(image_2)
        plt.show()
        plt.imshow(np.abs(image_1-image_2))
        plt.show()
    return image_1 - image_2 

def processing_gabor_filters(ksize = 8,sigma = 4,lambda_ = 8,gamma = 1):
    """
    gengerate 4 gabor filters
    """
    kernel_0   = cv2.getGaborKernel((ksize,ksize),sigma,0,lambda_,gamma,0)
    kernel_45  = cv2.getGaborKernel((ksize,ksize),sigma,np.pi/4 ,lambda_,gamma,0)
    kernel_90  = cv2.getGaborKernel((ksize,ksize),sigma,np.pi/2 ,lambda_,gamma,0)
    kernel_135 = cv2.getGaborKernel((ksize,ksize),sigma,np.pi/4*3,lambda_,gamma,0)

    return kernel_0, kernel_45, kernel_90, kernel_135

def gabor_filter(img,k0,k45,k90,k135):
   """
   filter convlution
   """
   return [cv2.filter2D(img, cv2.CV_16SC1, k0),cv2.filter2D(img, cv2.CV_16SC1, k45),cv2.filter2D(img, cv2.CV_16SC1, k90),cv2.filter2D(img, cv2.CV_16SC1, k135)]

def Is_scale(img_lst,c,s):
   return np.abs(subtraction(img_lst[c],img_lst[s]))

def RG_scale(Rs,Gs,c,s):
   return np.abs(subtraction(Rs[c]-Gs[c],Gs[s]-Rs[s]))

def BY_scale(Bs,Ys,c,s):
   return np.abs(subtraction(Bs[c]-Ys[c],Ys[s]-Bs[s]))

def O_c_s_theta(Os,c,s,theta):
       return np.abs(subtraction(Os[c][theta//45], Os[s][theta//45]))

def normalize_img(img,M=1):
    """
    normalize img scale to 0~M, globally multiply it by (M-\\bar{m})^2
    """
    image = img / np.max(img) * M if np.max(img) else img/10
    w,h = image.shape
    maxima = maximum_filter(image, size=(w/5,h/5))
    maxima = (image == maxima)
    mnum = maxima.sum()
    maxima = np.multiply(maxima, image)
    mbar = float(maxima.sum()) / mnum if mnum else 0
    return image*((M - mbar)**2)

def addition(img1,img2,shape):
    """
    through calculate we have fourth shape
    """
    image1 = cv2.resize(img1,shape)
    image2 = cv2.resize(img2,shape)
    return subtraction(image1, -image2)

def read_image(image_path):
    image = cv2.imread(image_path)
    image = image.astype("float32")
    return image

def find_maximum(image_dict):
    """
    find maximum
    """
    maximum = -1
    for key in image_dict.keys():
        maximum = max(maximum,np.max(image_dict[key]))  
    return maximum

def show_intensity_map(I_dict):
    I_maxima = find_maximum(I_dict)
    fig,ax = plt.subplots(2,3)
    ax[0,0].imshow(I_dict[2,5]/I_maxima,norm=None)
    ax[0,1].imshow(I_dict[3,6]/I_maxima,norm=None)
    ax[0,2].imshow(I_dict[4,7]/I_maxima,norm=None)
    ax[1,0].imshow(I_dict[2,6]/I_maxima,norm=None)
    ax[1,1].imshow(I_dict[3,7]/I_maxima,norm=None)
    ax[1,2].imshow(I_dict[4,8]/I_maxima,norm=None)
    plt.suptitle("6 * Intensity map")
    plt.show()

def show_colored_map(RG_dict,BY_dict):
    RG_maxima, BY_maxima = find_maximum(RG_dict),find_maximum(BY_dict)
    color_maxima = max(RG_maxima,BY_maxima)
    fig,ax = plt.subplots(4,3)
    ax[0,0].imshow(RG_dict[2,5]/color_maxima,norm=None)
    ax[0,1].imshow(RG_dict[3,6]/color_maxima,norm=None)
    ax[0,2].imshow(RG_dict[4,7]/color_maxima,norm=None)
    ax[1,0].imshow(RG_dict[2,6]/color_maxima,norm=None)
    ax[1,1].imshow(RG_dict[3,7]/color_maxima,norm=None)
    ax[1,2].imshow(RG_dict[4,8]/color_maxima,norm=None)
    ax[2,0].imshow(BY_dict[2,5]/color_maxima,norm=None)
    ax[2,1].imshow(BY_dict[3,6]/color_maxima,norm=None)
    ax[2,2].imshow(BY_dict[4,7]/color_maxima,norm=None)
    ax[3,0].imshow(BY_dict[2,6]/color_maxima,norm=None)
    ax[3,1].imshow(BY_dict[3,7]/color_maxima,norm=None)
    ax[3,2].imshow(BY_dict[4,8]/color_maxima,norm=None)
    plt.suptitle("12 * Color map, above 6 RG and down 6 BY")
    plt.show()

def show_orientation_map(O_dict):
    orientation_maxima = find_maximum(O_dict)
    fig,ax = plt.subplots(4,6)
    ax[0,0].imshow(O_dict[2,5,0]/orientation_maxima,norm=None)
    ax[0,1].imshow(O_dict[3,6,0]/orientation_maxima,norm=None)
    ax[0,2].imshow(O_dict[4,7,0]/orientation_maxima,norm=None)
    ax[0,3].imshow(O_dict[2,5,0]/orientation_maxima,norm=None)
    ax[0,4].imshow(O_dict[3,6,0]/orientation_maxima,norm=None)
    ax[0,5].imshow(O_dict[4,7,0]/orientation_maxima,norm=None)
    ax[1,0].imshow(O_dict[2,5,45]/orientation_maxima,norm=None)
    ax[1,1].imshow(O_dict[3,6,45]/orientation_maxima,norm=None)
    ax[1,2].imshow(O_dict[4,7,45]/orientation_maxima,norm=None)
    ax[1,3].imshow(O_dict[2,5,45]/orientation_maxima,norm=None)
    ax[1,4].imshow(O_dict[3,6,45]/orientation_maxima,norm=None)
    ax[1,5].imshow(O_dict[4,7,45]/orientation_maxima,norm=None)
    ax[2,0].imshow(O_dict[2,5,90]/orientation_maxima,norm=None)
    ax[2,1].imshow(O_dict[3,6,90]/orientation_maxima,norm=None)
    ax[2,2].imshow(O_dict[4,7,90]/orientation_maxima,norm=None)
    ax[2,3].imshow(O_dict[2,5,90]/orientation_maxima,norm=None)
    ax[2,4].imshow(O_dict[3,6,90]/orientation_maxima,norm=None)
    ax[2,5].imshow(O_dict[4,7,90]/orientation_maxima,norm=None)
    ax[3,0].imshow(O_dict[2,5,135]/orientation_maxima,norm=None)
    ax[3,1].imshow(O_dict[3,6,135]/orientation_maxima,norm=None)
    ax[3,2].imshow(O_dict[4,7,135]/orientation_maxima,norm=None)
    ax[3,3].imshow(O_dict[2,5,135]/orientation_maxima,norm=None)
    ax[3,4].imshow(O_dict[3,6,135]/orientation_maxima,norm=None)
    ax[3,5].imshow(O_dict[4,7,135]/orientation_maxima,norm=None)
    plt.suptitle("Orientation maps. each row i represents degree 45 * i, 0 degree represent | bar.")
    plt.show()

##--------------------------##
"""
the following function is the most important function
"""
##--------------------------##

def Itti_Saliency_map(image_path = "./test_images/standard.jpg", ifshow = False):
    print("reading image")
    try:
        image = read_image(image_path)
    except:
        if image_path.split(".")[-1] == "png":
            raise RuntimeError(f"Check if the path is correct ({image_path}) or change the png file into jpg format for it was wrongly saved")
        raise RuntimeError(f"Check if the path is correct ({image_path})")
    
    print("start processing")
        
    resized_image = resize_to_normal_shape(image)
    gaussian_img_list = eight_pyrimid_built(resized_image)

    b, g, r = seperate_RGB_chanells(resized_image)

    r_sigma = eight_pyrimid_built(r)
    g_sigma = eight_pyrimid_built(g)
    b_sigma = eight_pyrimid_built(b)
    I = [(r_sigma[i]+g_sigma[i]+b_sigma[i])/3 for i in range(9)]
    maximum = [np.max(I[i]) for i in range(9)]


    b = [np.where(b_sigma[i]>= 0.1 * maximum[i],b_sigma[i],0) for i in range(9)]
    g = [np.where(g_sigma[i]>= 0.1 * maximum[i],g_sigma[i],0) for i in range(9)]
    r = [np.where(r_sigma[i]>= 0.1 * maximum[i],r_sigma[i],0) for i in range(9)]

    Is = I
    Rs = [r[i]-(g[i]+b[i])/2 for i in range(9)]
    Gs = [g[i]-(r[i]+b[i])/2 for i in range(9)]
    Bs = [b[i]-(g[i]+r[i])/2 for i in range(9)]
    Ys = [(r[i]+g[i])/2 - np.abs(r[i] - g[i])/2 - b[i] for i in range(9)]


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

    print(f"we get 42 = {len(I_dict)+len(RG_dict)+len(BY_dict)+len(O_dict)} maps, including {len(I_dict)} Intensity maps,\
    {len(RG_dict)}+{len(BY_dict)} = {len(RG_dict)+len(BY_dict)} color maps and {len(O_dict)} orientation maps")
    if ifshow:
        show_intensity_map(I_dict)
        show_colored_map(RG_dict,BY_dict)
        show_orientation_map(O_dict)

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
    if ifshow:
        plt.imshow(I_bar)
        plt.title("I_bar")
        plt.show()

        plt.imshow(C_bar)
        plt.title("C_bar")
        plt.show()

        plt.imshow(O_bar)
        plt.title("O_bar")
        plt.show()
        plt.imshow(S)
        plt.title("Saliency map")
        plt.show()

        plt.imshow(normalize_img(S),cmap="gray")
        plt.title("Saliency map")
        plt.show()

    return S


if __name__ == "__main__":
    itti_saliency_map = Itti_Saliency_map("test_jpgs/mixed bars.jpg",ifshow=True)
    print(itti_saliency_map)
    plt.imshow(itti_saliency_map)
    plt.show()