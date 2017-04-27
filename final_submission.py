# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 03:43:37 2017

@author: che
"""
import cv2
from os import listdir,chdir
from matplotlib import pyplot as plt
import numpy as np
import pickle
from moviepy.editor import VideoFileClip
import math 



#set the working directory
chdir("C:\\Users\\che\\CarND-Advanced-Lane-Lines\\")
#path for camera calibration Images
path = "C:\\Users\\che\\CarND-Advanced-Lane-Lines\\camera_cal\\"
#list of filenames of camera calibration images
filename = listdir(path)

# set the number of inner corners
nx= 9
ny = 6

# prepare object points
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.


#Read the calibration images, find and draw the corners.
#save the images to file
for names in filename:
    img = cv2.imread(path+names)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,corners = cv2.findChessboardCorners(gray,(nx,ny),None)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        write_name = 'corners_found'+names+'.jpg'
        cv2.imwrite(write_name, img)
        cv2.imshow('img', img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

#Path for test images
test_path = "C:\\Users\\che\\CarND-Advanced-Lane-Lines\\test_images\\"
#create list of names of test images
file_name = listdir(test_path)

# Read the calibration images, undistort, view and save them to file
for names in filename:
    img = cv2.imread(path+names)
    img_size = (img.shape[1], img.shape[0])
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=8)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=8)
    undistort_name = 'undistort'+names+'.jpg'
    plt.savefig(undistort_name)


# Read the test images, undistort, view and save them to file
for names in file_name:
    img = cv2.imread(test_path+names)
    img_size = (img.shape[1], img.shape[0])
    write_name = 'original_image'+names+'.jpg'
    cv2.imwrite(write_name,img)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=8)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=8)
    undistort_name = 'undistort'+names+'.jpg'
    cv2.imwrite(undistort_name,dst)


#serialize and save the required metrics
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "wide_dist_pickle.p", "wb" ))

#Function for returning a combination of thresholded
#gradient and s-channel (hls) binary image
def pipeline (img,s_thresh=(0,255),sx_thresh=(0,255),sobel_kernel=3):
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel) 
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    abs_sobelx = (sobelx**2)
    abs_sobely = (sobely**2)
    abs_sobelxy = np.sqrt(abs_sobelx+abs_sobely)
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy)) #standardisation
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & 
             (scaled_sobel <= sx_thresh[1])] = 1 # Threshold of gradient
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & 
             (s_channel <= s_thresh[1])] = 1 #Threshold for s_channel
    color_binary = np.dstack((np.zeros_like(sxbinary),sxbinary,s_binary))
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary ==1)] = 1 #Binary image of gradient and color threshold
    return combined_binary

#Function for returning a combination of thresholded
#gradient and s-channel (hls) binary image
#The gradient and s_channel thresholds will be displayed in different colors
def pipeline2 (img,s_thresh=(0,255),sx_thresh=(0,255),sobel_kernel=3):
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel) 
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    abs_sobelx = (sobelx**2)
    abs_sobely = (sobely**2)
    abs_sobelxy = np.sqrt(abs_sobelx+abs_sobely)
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy)) #standardisation
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & 
             (scaled_sobel <= sx_thresh[1])] = 1 # Threshold of gradient
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & 
             (s_channel <= s_thresh[1])] = 1 #Threshold for s_channel
    color_binary = np.dstack((np.zeros_like(sxbinary),sxbinary,s_binary))
    return color_binary


#Read the images in test_images folder
#convert the images to binary format with gradient and s_channel thresholding
#Display and save the images to file 
for names in file_name:
    img = cv2.imread(test_path+names)
    img_size = (img.shape[1], img.shape[0])
    write_name = 'original_image'+names+'.jpg'
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    binary_image = pipeline(dst,sobel_kernel=3,sx_thresh=(20,100),s_thresh=(70,255))
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=8)
    ax2.imshow(binary_image)
    ax2.set_title('binary_image', fontsize=8)
    binary_name = 'binary_'+names+'.jpg'
    plt.savefig(binary_name)
    
#Read the images in test_images folder
#convert the images to binary format with gradient and s_channel thresholding
#The gradient and s_channel thresholds will be displayed in different colors
for names in file_name:
    img = cv2.imread(test_path+names)
    img_size = (img.shape[1], img.shape[0])
    write_name = 'original_image'+names+'.jpg'
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    color_binary = pipeline2(dst,sobel_kernel=3,sx_thresh=(20,100),s_thresh=(70,255))
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=8)
    ax2.imshow(color_binary)
    ax2.set_title('color_binary', fontsize=8)
    color_name = 'color_'+names+'.jpg'
    plt.savefig(color_name)

#set source and destination points for perspective transform
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
M = cv2.getPerspectiveTransform(src,dst)
warped = cv2.warpPerspective(binary, M, img_size)
plt.imshow(warped)

#Read the images in test_images folder
#convert the images to binary format with gradient and s_channel thresholding
#perform perspective transform on the image
#Display and save the images to file 
for names in file_name:
    img = cv2.imread(test_path+names)
    img_size = (img.shape[1], img.shape[0])
    write_name = 'original_image'+names+'.jpg'
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    binary_image = pipeline(dst,sobel_kernel=3,sx_thresh=(20,100),s_thresh=(70,255))
    warped_binary = cv2.warpPerspective(binary_image, M, img_size)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=8)
    ax2.imshow(warped_binary)
    ax2.set_title('warped_binary', fontsize=8)
    warped_name_binary = 'warped_binary_'+names+'.jpg'
    plt.savefig(warped_name_binary)

#Display the undistorted image with source points and perspective warped
#image with source and destination points 
for names in file_name:
    img = cv2.imread(test_path+names)
    img_size = (img.shape[1], img.shape[0])
    write_name = 'original_image'+names+'.jpg'
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    #binary_image = pipeline(dst,sobel_kernel=3,sx_thresh=(20,100),s_thresh=(70,255))
    warped_binary = cv2.warpPerspective(dst, M, img_size)
    red = (255,0,90)
    green = (255,255,255)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
    cv2.line(dst, (203, 720), (585, 460),red,5)
    cv2.line(dst,(585,460),(695,460),red,5)
    cv2.line(dst,(695,460),(1127,720),red,5)
    cv2.line(dst,(1127,720),(203,720),red,5)
    ax1.imshow(dst)
    ax1.set_title('undistorted Image', fontsize=8)
    dist_scr_dst_name = 'undistorted_src_dst'+names+'.jpg'
    cv2.imwrite(dist_scr_dst_name,dst)
    #cv2.line(warped_binary, (203, 720), (585, 460),red,5)
    #cv2.line(warped_binary,(585,460),(695,460),red,5)
    #cv2.line(warped_binary,(695,460),(1127,720),red,5)
    #cv2.line(warped_binary,(1127,720),(203,720),red,5)
    cv2.line(warped_binary, (320, 720), (320, 0),red,5)
    cv2.line(warped_binary,(960,0),(960,720),red,5)
    ax2.imshow(warped_binary)
    ax2.set_title('warped_binary', fontsize=8)
    warped_name = 'warped_binary_src_dst'+names+'.jpg'
    plt.savefig(warped_name)
    
#Function for finding the lanelines using sliding window method for the first image
#From the second image onwards the search for lanelines will be done only in a 
#margin +/_ to the polynomial fit of the first image
i = 1
def drawlanelines(original_image,binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0) #Histogram for the bottom half of image
    out_img = np.dstack((binary_warped,binary_warped,binary_warped))*255 #Image with white lanelines
    midpoint = np.int(histogram.shape[0]/2) #Find the midpoint of the image
    leftx_base = np.argmax(histogram[:midpoint]) #position of left lane on x-axis 
    rightx_base = np.argmax(histogram[midpoint:])+midpoint #position of right lane on x-axis
    nwindows = 9 #No of windows
    window_height = np.int(binary_warped.shape[0]/nwindows) #Heifht of each window
    nonzero = binary_warped.nonzero() #Number of non-zero values in binary_warped image
    nonzeroy = np.array(nonzero[0]) 
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100 #Margin on either side of the windows
    minpix = 50 #pixel size for adjusting the window according to change in position of lane lines
    left_lane_inds = [] 
    right_lane_inds = [] 
    global i
    if i == 1: #condition which will be satisfied by the first image in a video
        for window in range(nwindows):
            #calculate the values for the window
            win_y_low = binary_warped.shape[0]-(window+1)*window_height
            win_y_high = binary_warped.shape[0]-window*window_height
            win_xleft_low = leftx_current-margin
            win_xleft_high = leftx_current+margin
            win_xright_low = rightx_current-margin
            win_xright_high = rightx_current+margin
            #Draw the windows
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,
                  win_y_high),(0,255,0),2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,
                  win_y_high),(0,255,0),2)
            # Find the non zero point within the windows
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            #Append the list with non-zero points found within the window
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            #Adjust the window position according to the position of lane lines
            if len(good_left_inds) > minpix:
                leftx_current= np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        global left_fit 
        global right_fit 
        #Fit the second order polynomial for the detected points on the left and right side
        left_fit = np.polyfit(lefty,leftx,2)
        right_fit = np.polyfit(righty,rightx,2)
        global ploty
        ploty = np.linspace(0, binary_warped.shape[0]-1, 
                binary_warped.shape[0])
        #Quadratic equation for the left and right lane lines
        left_fitx = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*+ploty+right_fit[2]
        i = i+1
        
    # This code will be run from the second image onwards which narrow downs the search
    else:
        # Reduce the search area for finding the lane lines on left and right
        # search within a margin around the polynomial fit of the first image
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2)+left_fit[1]*nonzeroy+
                   left_fit[2]-margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2)+
                           left_fit[1]*nonzeroy+left_fit[2]+margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2)+right_fit[1]*nonzeroy+
                                right_fit[2]-margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2)+
                                right_fit[1]*nonzeroy+right_fit[2]+margin)))
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx,2)
        right_fit = np.polyfit(righty,rightx,2)
        left_fitx = left_fit[0]*ploty**2 +left_fit[1]*ploty+left_fit[2]
        right_fitx = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
    
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds],nonzerox[left_lane_inds]] = [255,0,0]
    out_img[nonzeroy[right_lane_inds],nonzerox[right_lane_inds]]=[0,0,255]
    # Recast the x and y points into usable format for cv2.fillPoly()
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    #create an inverse of the perspective matrix
    Minv = np.linalg.inv(M)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (original.shape[1], original.shape[0])) 
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    y_eval = np.max(ploty)
    # Calculate the new radii of curvature
    left_curverad = (((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0]))
    #left_curverad.astype(int)
    right_curverad = (((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0]))
    #Average of the left and right radius for displaying on the image
    radius = (left_curverad+right_curverad)/2
    #calculate the center of the image in meters
    center_of_image = 3.4/640
    #calculate the lane center in pixels and convert it into meters
    lane_center_pixels = ([np.sqrt((right_fitx-left_fitx)**2)/2])
    lane_center = 1.85/np.mean(lane_center_pixels)
    #calculate the offset for displaying on the image
    offset = center_of_image - lane_center
    # Overlay the Radius of curvature and offset on the image
    cv2.putText(original_image, "Radius of curvature: %.3f m" %(radius), 
        (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(original_image, "offset to the left: %.8f m" % (offset), 
        (90,90), cv2.FONT_HERSHEY_SIMPLEX, 1, 2) 
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    return result

#create a final pipeline for the images which includes undistortion, gradient and 
#color thresholding, creating binary image, perspective warping , finding lane lines
   
def final_pipeline (image):
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    binary = pipeline(undist,sobel_kernel=3,sx_thresh=(20,100),s_thresh=(70,255))
    perspective_warped = cv2.warpPerspective(binary,M,img_size)
    final_result = drawlanelines(image,perspective_warped)
    return final_result

for names in file_name:
    img = cv2.imread(test_path+names)
    img_size = (img.shape[1], img.shape[0])
    final_output = final_pipeline(img)
    write_name = 'original_image'+names+'.jpg'
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=8)
    ax2.imshow(final_output)
    ax2.set_title('final_output', fontsize=8)
    final_image = 'final_output'+names+'.jpg'
    plt.savefig(final_image)

    
#Read the video frame by frame. Apply the pipeline and write the video to file    
white_output = 'submission.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(final_pipeline) 
%time white_clip.write_videofile(white_output, audio=False)

