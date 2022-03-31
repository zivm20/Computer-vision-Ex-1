"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 209904606


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    try:
        img = cv2.cvtColor(cv2.imread(filename,1), cv2.COLOR_BGR2RGB)
    except:
        return np.zeros( (256,256,3),np.float32)
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img/255

    
    


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    
    img = imReadAndConvert(filename,representation)
    if representation == 1:
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    #ensure image has 3 channels
    try:
        ensure3ChannelImgInput(imgRGB)
    except Exception as e:
        print(e)
        return imgRGB

    #YIQ conversion matrix
    YIQ_mat = np.array([[ 0.299, 0.587, 0.114],
                        [ 0.596, -0.275, -0.321],
                        [ 0.212, -0.523, 0.311]])
    
    #turn our NxMx3 matrix to (N*M)x3 matrix
    img_vals = imgRGB.reshape((imgRGB.shape[0] * imgRGB.shape[1], 3))

    #matrix multiplication to convert each RGB triplet into YIQ
    img_vals = np.matmul(img_vals,YIQ_mat)

    #reshape our matrix back
    return img_vals.reshape(imgRGB.shape).astype(np.float32)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    #ensure image has 3 channels
    try:
        ensure3ChannelImgInput(imgYIQ)
    except Exception as e:
        print(e)
        return imgYIQ

    #YIQ conversion matrix
    YIQ_mat = np.array([[ 0.299, 0.587, 0.114],
                        [ 0.596, -0.275, -0.321],
                        [ 0.212, -0.523, 0.311]])
    #RGB conversion matrix is the inverse of the YIQ conversion matrix
    RGB_mat = np.linalg.inv(YIQ_mat)

    #turn our NxMx3 matrix to (N*M)x3 matrix
    img_vals = imgYIQ.reshape((imgYIQ.shape[0] * imgYIQ.shape[1], 3))

    #matrix multiplication to convert each YIQ triplet into RGB
    img_vals = np.matmul(img_vals,RGB_mat)

    #reshape our matrix back
    return img_vals.reshape(imgYIQ.shape).astype(np.float32)

    


def hsitogramEqualize(imgOrig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    img = imgOrig
    #handle case for inputing RGB
    if len(imgOrig.shape) == 3:
        try:
            ensure3ChannelImgInput(imgOrig)
        except Exception as e:
            print(e)
            return
        #get only the Y channel of our RGB image converted to YIQ
        img = transformRGB2YIQ(img)
        imEq, histOrg, histEq = handle_hist(img[:,:,0])
        img[:,:,0] = imEq
        
        img = transformYIQ2RGB(img)
        return img,histOrg,histEq
    #handle case for inputing image that isn't grayscale or RGB
    elif len(imgOrig.shape) != 2:
        print("Image must be RGB or grayscale!")
        return
    
    return handle_hist(img)

    


def handle_hist(img):
    #will do histogram equalization on some color channel 
    histOrg,bins = np.histogram(img*255,bins=256,range=[0,255])
    cumSum = np.cumsum(histOrg)
    imEq = img*255   
    
    for i in range(256):
        lut =  np.ceil(255.0*float(cumSum[i])/float(cumSum[-1]))
        imEq[np.logical_and(img*255>=bins[i],img*255<bins[i+1])] = lut
        #private case where the last bin is for values in range [254,255] instead of [254,255)
        if i==255:
            imEq[np.logical_and(img*255>=bins[i],img*255==bins[i+1])] = lut
    
    histEq,_ = np.histogram(imEq,bins=256,range=[0,255])
    imEq = imEq/255

    return imEq, histOrg, histEq


    



def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> Tuple[List[np.ndarray], List[float]]:
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass

def ensure3ChannelImgInput(img:np.ndarray):
    if len(img.shape) !=3 or img.shape[2] != 3:
        raise ValueError("Given image doesn't have 3 channels")

if __name__=="__main__":
    f = "yiq_example.jpeg"
    """
    f = "sjhfjkSBFP:KH; p"
    
    a = imReadAndConvert(f,1)
    b = imReadAndConvert(f,2)
    cv2.imshow('a',a)
    cv2.imshow('b',b)
    

    imDisplay(f,1)
    imDisplay(f,2)
    print(b)
    cv2.waitKey(0)
    """

    a = imReadAndConvert(f,2)
    a2 = transformRGB2YIQ(a)
    a3 = transformYIQ2RGB(a2)
    plt.imshow(a)
    plt.show()
    plt.imshow(a2)
    plt.show()
    plt.imshow(a3)
    plt.show()
    

    img=np.array([[ [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3] ],
              [ [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3] ],
              [ [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3] ],
              [ [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3] ],
              [ [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3] ],
              [ [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3] ],
              [ [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3] ],
              [ [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3] ]])
    print(img[:,:,0])
    print(img.shape)
    print(img[:,:,0].shape)

