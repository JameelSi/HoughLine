import numpy as np
import cv2
from matplotlib import pyplot as plt
import ex3a as funcs

def hough_line(img):
    # Rho and Theta ranges
    #changed -90 to 0
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int( np.ceil(np.sqrt(width * width + height * height)) ) # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img) # (row, col) indexes to edges
    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int( round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len )
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos
def houghTransform(img,accumulator, thetas, rhos):
    xs=list()
    # since the question wants the most strong 4 
    for i in range(8):
        idx = np.argmax(accumulator)
        rrho=idx // accumulator.shape[1]
        rho = rhos[rrho]  # distance from origin
        ttheta=idx % accumulator.shape[1]
        theta = thetas[ttheta]  # angle
        # finding the x,y coordinates
        cos = np.cos(theta)
        sin = np.sin(theta)
        x0 = int(cos*rho)
        y0 = int(sin*rho)
        #adjusting the coordinate of the square 
        x1=int(x0 - 120 * sin)
        y1=int(y0 + 70 * cos)
        x2=int(x0 - 280 * sin)
        y2=int(y0 + 330 * cos)
        cv2.line(img, (x1,y1), (x2,y2), (255,255,255), 2)
        # marking the max as 0 (same as poping it) so that 2nd max become max
        accumulator[rrho, ttheta] = 0

if __name__ == "__main__":
    image = np.zeros((400,400))
    cv2.rectangle(image,(120,70),(280,330),(255,255,255),-1)
    #get image mean and std
    mean,std = cv2.meanStdDev(image)
    #adding noise  
    noise = np.random.normal(mean/5,std/5, (400, 400))
    #apply the noise 
    noisy =image+noise
    #normalize it 
    noisy = (255*(noisy - np.min(noisy))/np.ptp(noisy)).astype(int)
    ######### Canny #########
    kernel = np.array([[1,4,6,4, 1], [4, 16, 24,16,4], [6, 24, 36,24,6],[4,16,24,16,4],[1,4,6,4,1]])/256
    # apply filter
    smoothy=funcs.convolve2d(noisy,kernel,5)
    # sobel filter
    ix,G,D=funcs.gradient_intensity(smoothy)
    # suppresion
    sup=funcs.suppression(G,D)
    # threshold
    sup,weak=funcs.threshold(sup,155,255)
    # tracking
    Canny=funcs.tracking(sup,weak)
    #########################
    # Hough Transform
    accumulator, thetas, rhos=hough_line(Canny)
    hough=np.copy(Canny)
    houghTransform(hough,accumulator, thetas, rhos)
    # Plot the results
    plt.subplots(1,1,figsize=(10,5))
    plt.subplot(1,4,1)
    plt.imshow(image, 'gray', vmin=0, vmax=255)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(noisy, 'gray', vmin=0, vmax=255)
    plt.title('Noisy image')
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.imshow(Canny, 'gray', vmin=0, vmax=255)
    plt.title('Canny applied')
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.imshow(hough, 'gray', vmin=0, vmax=255)
    plt.title('Hough Transform')
    plt.axis('off')

    plt.tight_layout()
    plt.show()