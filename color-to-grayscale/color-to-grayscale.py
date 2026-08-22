import numpy as np
def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    img=np.array(image)
    grayscale_list=[]
    for i in range(len(img)):
        row = []
        for j in img[i]:
            y=0.299*j[0]+0.587*j[1]+0.114*j[2]
            row.append(y)
        grayscale_list.append(row)    
    return grayscale_list 