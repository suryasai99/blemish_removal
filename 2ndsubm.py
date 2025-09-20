import cv2
import matplotlib.pyplot as plt
import numpy as np

r = 15
image = cv2.imread('/Users/suryasaikadali/Downloads/open_Cv/CVIP/M5/blemish.png', 1)

# This function computes the mean of the laplacian gradient inside a circular mask
def laplacianfilter(x,y):
    global image,r
    # Convert to grayscale
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create circular mask
    mask = np.zeros_like(img_grey, dtype=np.uint8)
    cv2.circle(mask, (x,y), r, 255, -1)

    # compute the laplacian gradient
    lap = cv2.Laplacian(img_grey, cv2.CV_32F, ksize = 3, scale = 1, delta = 0)

    # Take absolute values 
    abs_lap = np.absolute(lap)

    # Compute mean only inside circular mask
    mean_grad = cv2.mean(abs_lap, mask=mask)[0]
    
    return mean_grad

# This function computes the best patch center from the 8 neighbours
def findbestpatch(x,y):
     # neighbour patch centers (8 neighbors around the circle)
    neighbours = [
        (x + 2*r, y),
        (x - 2*r, y),
        (x, y + 2*r),
        (x, y - 2*r),
        (x + 2*r, y + 2*r),
        (x - 2*r, y + 2*r),
        (x + 2*r, y - 2*r),
        (x - 2*r, y - 2*r)
    ]
    # List to store mean gradients for each patch
    gradients = []
    for (cx, cy) in neighbours:
        mean_grad = laplacianfilter(cx, cy)
        gradients.append((mean_grad, cx, cy))

    min_grad, best_x, best_y = min(gradients, key=lambda item: item[0])
    return min_grad, best_x, best_y

# This function crops the square patch and creates the circular mask
def crop_region(x,y):
    # Compute bounding box of the circle
    x1, y1 = max(x - r, 0), max(y - r, 0)
    x2, y2 = min(x + r, image.shape[1]), min(y + r, image.shape[0])

    # Crop the square patch
    patch = image[y1:y2, x1:x2].copy()

    # Create a mask of same size
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.circle(mask, (x - x1, y - y1), r, 255, -1)

    return patch, mask

# Mouse callback function
def mouseaction(event,x, y, flags, userdata):
    global image
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the center of the blemish
        blemish_centre = (x,y)

        # Find the best patch
        _, best_x, best_y = findbestpatch(x,y)

        # Crop the source patch and create mask
        src_img, mask = crop_region(best_x, best_y)

        # Perform seamless cloning
        output = cv2.seamlessClone(src_img, image, mask, blemish_centre, cv2.NORMAL_CLONE)

        # Update global image so multiple edits accumulate
        image[:] = output  
        cv2.imshow("blemish", output)
    
    

cv2.namedWindow("blemish")
cv2.setMouseCallback("blemish", mouseaction)

original = image.copy()

while True:
    cv2.imshow('blemish', image)
    k = cv2.waitKey(1) & 0XFF

    if k == 27: # esc to quit
        break
    elif k==115: # To save the image by pressing 's'
        cv2.imwrite("edited_image.png", image)
    elif k==114: # To reset the image by pressing 'r'
        image = original.copy()

cv2.destroyAllWindows()