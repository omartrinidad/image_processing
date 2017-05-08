import cv2
import numpy as np

def mouse_handler(event, x, y, flags, data) :
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),3, (0,0,255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < 4 :
            data['points'].append([x,y])

def get_four_points(im):
    
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    
    #Set the callback function for any mouse event
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    
    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)
    
    return points



if __name__ == '__main__' :

    # Read source image.
    im_src = cv2.imread('clock.jpg');
    size = im_src.shape
   
    # Create a vector of source points.
    pts_src = np.array(
                       [
                        [0,0],
                        [size[1] - 1, 0],
                        [size[1] - 1, size[0] -1],
                        [0, size[0] - 1 ]
                        ],dtype=float
                       );

    
    # Read destination image
    im_dst = cv2.imread('isle.jpg');
    # Get four corners of the billboard
    
   
    pts_dst = get_four_points(im_dst)
    # or just define the points
        #pts_dst =  np.array([[215, 56],[365, 10],[364, 296],[218, 258]])
    
    # Calculate Homography between source and destination points
    h, status = cv2.findHomography(pts_src, pts_dst);
    
    # Warp source image
    im_temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

    cv2.imshow("Imag2e", im_temp);
    # Black out polygonal area in destination image.
    cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16);
    cv2.imshow("Imasdage", im_dst);

    # Add warped source image to destination image.
    im_dst = im_dst + im_temp;
    
    # Display image.
    cv2.imshow("Image", im_dst);
    cv2.waitKey(0);
