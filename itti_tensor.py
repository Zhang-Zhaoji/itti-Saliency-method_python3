import cv2
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import maximum_filter
import torch.nn.functional as F
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

def gaussian_pyrimid(image: torch.Tensor) -> torch.Tensor:
   """
   return /2 resolution gaussian pyrimid\n
   image: torch.tensor
   """
   #image = transforms.GaussianBlur((3,3),(1,1))(image)
   image_shape = image.shape
   if len(image_shape) == 3:
       image = image.unsqueeze(0)
       image_shape = image.shape
   down_sampling_image = F.adaptive_avg_pool2d(image,(image_shape[2]//2+1,image_shape[3]//2+1))
   return down_sampling_image

def eight_pyrimid_built(image: torch.Tensor):
   """
   yielding horizontal and vertical image-reduction factors ranging from 1:1 (scale zero) to 1:256 (scale eight) in eight octaves.
   """
   image_list = []
   image2 = image.clone()
   for i in range(9):
      image_list.append(image2)
      image2 = gaussian_pyrimid(image2)
   return image_list

def resize_to_normal_shape(image: torch.Tensor):
    """
    Input is provided in the form of static color images, usually digitized at 640*480 resolution for ndarray, but [batch = 1,channel = 3,480,640] for tensor.
    """
    resized_image = F.interpolate(image.unsqueeze(0),size=(480,640),mode="bilinear")
    return resized_image

def seperate_RGB_chanells(image:torch.Tensor):
   """
    Seperate RGB chanells to going through following algorithms
    outcome is r,g,b for 3 torch.Tensor shape = [1,1,H,W]
    """
   return image.unbind(1)


def subtraction(img1: torch.Tensor,img2: torch.Tensor,ifshow = False):
    """
    return img1 - img2 but resized to the higher resolution
    """
    if ifshow:
        # not good for tensors.
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
    if len(img1.shape)== 3:
        img1.unsqueeze(0)
    if len(img2.shape)== 3:
        img2.unsqueeze(0)
    shape = img1.shape if sum(img1.shape) >= sum(img2.shape) else img2.shape
    image_1 = F.interpolate(img1,shape[2:],mode="bilinear")
    image_2 = F.interpolate(img2,shape[2:],mode="bilinear")
    if ifshow:
        plt.imshow(image_2)
        plt.show()
        plt.imshow(torch.abs(image_1-image_2))
        plt.show()
    return image_1 - image_2 

def processing_gabor_filters(ksize = 8,sigma = 4,lambda_ = 8,gamma = 1):
    """
    gengerate 4 gabor filters
    """
    kernel_0   = cv2.getGaborKernel((ksize,ksize),sigma,0,lambda_,gamma,0)
    kernel_45  = cv2.getGaborKernel((ksize,ksize),sigma,torch.pi/4 ,lambda_,gamma,0)
    kernel_90  = cv2.getGaborKernel((ksize,ksize),sigma,torch.pi/2 ,lambda_,gamma,0)
    kernel_135 = cv2.getGaborKernel((ksize,ksize),sigma,torch.pi/4*3,lambda_,gamma,0)

    return kernel_0, kernel_45, kernel_90, kernel_135

def gabor_filter(img,k0,k45,k90,k135):
   """
   filter convlution
   """
   device = "cuda" if torch.cuda.is_available() else "cpu"
   if len(img.shape) == 3:
       img = img.unsqueeze(0)
   k0,k45,k90,k135 = map(lambda x: torch.reshape(torch.tensor(x,dtype=torch.float32),(1,1,len(k0),len(k0))),[k0,k45,k90,k135])
   k0,k45,k90,k135 = map(lambda x:x.to(device),[k0,k45,k90,k135])
   return list(map(lambda x: F.conv2d(img,x,padding=x.shape[2]//2),[k0,k45,k90,k135]))
   # return [cv2.filter2D(img, cv2.CV_16SC1, k0),cv2.filter2D(img, cv2.CV_16SC1, k45),cv2.filter2D(img, cv2.CV_16SC1, k90),cv2.filter2D(img, cv2.CV_16SC1, k135)]

def Is_scale(img_lst,c,s):
   return torch.abs(subtraction(img_lst[c],img_lst[s]))

def RG_scale(Rs,Gs,c,s):
   return torch.abs(subtraction(Rs[c]-Gs[c],Gs[s]-Rs[s]))

def BY_scale(Bs,Ys,c,s):
   return torch.abs(subtraction(Bs[c]-Ys[c],Ys[s]-Bs[s]))

def O_c_s_theta(Os,c,s,theta):
       return torch.abs(subtraction(Os[c][theta//45], Os[s][theta//45]))

def normalize_img(img,M=1):
    """
    normalize img scale to 0~M, globally multiply it by (M-\\bar{m})^2
    """
    eps = 1e-8
    image = img / torch.max(img) * M if torch.max(img) != 0 else img * 0
    image_calculate = image[0,0,:,:].cpu()
    image_calculate = image_calculate.numpy()
    w,h = image_calculate.shape
    maxima = maximum_filter(image_calculate, size=(w/5,h/5))
    maxima = (image_calculate == maxima)
    mnum = maxima.sum()
    maxima = maxima * image_calculate
    mbar = float(maxima.sum()) / (mnum+eps) if mnum else 0
    return image*((M - mbar)**2)

def addition(img1,img2,shape):
    """
    through calculate we have fourth shape
    """
    # print(img1.shape)
    image1 = F.interpolate(img1,shape,mode="bilinear")
    image2 = F.interpolate(img2,shape,mode="bilinear")
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
        maximum = max(maximum,torch.max(image_dict[key]))  
    return maximum

##--------------------------##
"""
the following function is the most important function
"""
##--------------------------##

def Itti_down_sampling(image, ifshow = False):
    resized_image = resize_to_normal_shape(image) # [1,3,480,640]
    r,g,b = seperate_RGB_chanells(resized_image) # [1,480,640]
    r_sigma = eight_pyrimid_built(r) # [1,1,H,W]
    g_sigma = eight_pyrimid_built(g) # [1,1,H,W]
    b_sigma = eight_pyrimid_built(b) # [1,1,H,W]

    I = [(r_sigma[i]+g_sigma[i]+b_sigma[i])/3 for i in range(9)]
    maximum = [torch.max(I[i]) for i in range(9)]
    b = [torch.where(b_sigma[i]>= 0.1 * maximum[i],b_sigma[i],0) for i in range(9)]
    g = [torch.where(g_sigma[i]>= 0.1 * maximum[i],g_sigma[i],0) for i in range(9)]
    r = [torch.where(r_sigma[i]>= 0.1 * maximum[i],r_sigma[i],0) for i in range(9)]

    Is = I
    Rs = [r[i]-(g[i]+b[i])/2 for i in range(9)]
    Gs = [g[i]-(r[i]+b[i])/2 for i in range(9)]
    Bs = [b[i]-(g[i]+r[i])/2 for i in range(9)]
    Ys = [(r[i]+g[i])/2 - torch.abs(r[i] - g[i])/2 - b[i] for i in range(9)]
    kernel_0, kernel_45, kernel_90, kernel_135 = processing_gabor_filters()
    Os = [gabor_filter(Is[i],kernel_0,kernel_45,kernel_90,kernel_135) for i in range(9)]
    return Is, Rs, Gs, Bs, Ys, Os

def Itti_feature_maps(Is,Rs,Gs,Bs,Ys,Os):
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
    return I_dict, RG_dict, BY_dict, O_dict

def Itti_motion_conspicuous_maps(fIbar,fCbar,fObar,cIbar,cCbar,cObar):
    """
    f for formal frame, and p for present frame
    we use Mp-Mf as a map to detect motion since Mp-Mf = \Delta M \times Velocity \times \Delta t 
    """
    # fIbar,fCbar,fObar,cIbar,cCbar,cObar = map(lambda x: x.astype(np.int16),[fIbar,fCbar,fObar,cIbar,cCbar,cObar])
    mI_bar, mC_bar, mO_bar = map(torch.abs,[fIbar-cIbar,fCbar-cCbar,fObar-cObar])
    # mI_bar, mC_bar, mO_bar = map(lambda x: x.astype(np.int8),[mI_bar, mC_bar, mO_bar])
    return mI_bar, mC_bar, mO_bar

def synthesis_conspicuous_map(I_dict, RG_dict, BY_dict, O_dict,addition_shape):
    # print(addition_shape)
    c_set = (2,3,4)
    delta_set = (3,4)
    theta_set = (0,45,90,135)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    I_bar = torch.zeros((1,1,1,1)).to(device)
    C_bar = torch.zeros((1,1,1,1)).to(device)
    O_bar_0 = torch.zeros((1,1,1,1)).to(device)
    O_bar_45 = torch.zeros((1,1,1,1)).to(device)
    O_bar_90 = torch.zeros((1,1,1,1)).to(device)
    O_bar_135 = torch.zeros((1,1,1,1)).to(device)
    for c in c_set:
        for delta in delta_set:
            # print("first place")
            I_bar = addition(I_bar,normalize_img(I_dict[(c,c+delta)]),addition_shape)
            C_bar = addition(C_bar,normalize_img(RG_dict[(c,c+delta)]),addition_shape)
            C_bar = addition(C_bar,normalize_img(BY_dict[(c,c+delta)]),addition_shape)

            O_bar_0 = addition(O_bar_0,normalize_img(O_dict[(c,c+delta,0)]),addition_shape)
            O_bar_45 = addition(O_bar_45,normalize_img(O_dict[(c,c+delta,45)]),addition_shape)
            O_bar_90 = addition(O_bar_90,normalize_img(O_dict[(c,c+delta,90)]),addition_shape)
            O_bar_135 = addition(O_bar_135,normalize_img(O_dict[(c,c+delta,135)]),addition_shape)
    O_bar = torch.zeros((1,1,1,1)).to(device)
    for O_bar_theta in [O_bar_0, O_bar_45, O_bar_90, O_bar_135]:
        O_bar = addition(O_bar,normalize_img(O_bar_theta),addition_shape)
    return I_bar, C_bar, O_bar

def Itti_conspicuous_maps(image, ifshow = False):
    Is, Rs, Gs, Bs, Ys, Os = Itti_down_sampling(image,ifshow)
    I_dict, RG_dict, BY_dict, O_dict =Itti_feature_maps(Is, Rs, Gs, Bs, Ys, Os)
    # print((Is[4].shape[2],Is[4].shape[3]))
    I_bar, C_bar, O_bar = synthesis_conspicuous_map(I_dict, RG_dict, BY_dict, O_dict,(Is[4].shape[2],Is[4].shape[3]))

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

    return I_bar, C_bar, O_bar
