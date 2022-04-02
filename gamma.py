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
from ex1_utils import LOAD_GRAY_SCALE
import ex1_utils as ex1
import cv2
import numpy as np
def empty(x):
    pass
def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    #read image and convert to bgr if needed and change the range to [0,255]
    imgOrg = ex1.imReadAndConvert(img_path,rep)
    imgOrg = (imgOrg*255).astype(np.uint8)
    if rep == 2:
        imgOrg = cv2.cvtColor(imgOrg,cv2.COLOR_RGB2BGR)

    #create window and gamma trackbar
    cv2.namedWindow("Gamma Correction")
    cv2.createTrackbar("gamma","Gamma Correction",100,200,empty)
    
    #save a copy of the original image to correct by it
    img = imgOrg.copy()
    gamma = -1

    while True:
        #exit
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        #get current positions of all Three trackbars
        g = cv2.getTrackbarPos('gamma', "Gamma Correction")
        if g != gamma:
            img = gammaCorrect(imgOrg,g)
            gamma = g 
            cv2.imshow("Gamma Correction", img)
       
    cv2.destroyAllWindows()
    
        
    
    
#gamma correct img
def gammaCorrect(img,gamma) -> np.ndarray:
    #case where gamma is 0
    if gamma == 0:
        gamma = 1
    #scale gamma by 100 to get the range of [0,2]
    gamma = gamma/100

    #create lookup table
    lut = np.array([ ((i/255)**(1/gamma))*255 for i in range(256)],np.uint8)
   
    return cv2.LUT(img,lut)




def main():
    gammaDisplay('beach.jpg', 2)


if __name__ == '__main__':
    main()
