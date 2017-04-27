import cv2
from os import listdir,chdir
from matplotlib import pyplot as plt
import numpy as np


chdir("C:\\Users\\che\\CarND-Advanced-Lane-Lines\\")
path = "C:\\Users\\che\\CarND-Advanced-Lane-Lines\\camera_cal\\"
filename = listdir(path)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

nx= 9
ny = 6

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


# Test undistortion on an image
img = cv2.imread(path+filename[0])
img_size = (img.shape[1], img.shape[0])
cv2.imwrite('output_images/original_1.jpg',img)

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)




#Thresholded Binary image
def pipeline (img,s_thresh=(0,255),sx_thresh=(0,255),sobel_kernel=3):
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    abs_sobelx = (sobelx**2)
    abs_sobely = (sobely**2)
    abs_sobelxy = np.sqrt(abs_sobelx+abs_sobely)
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & 
             (scaled_sobel <= sx_thresh[1])] = 1
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & 
             (s_channel <= s_thresh[1])] = 1
    #color_binary = np.dstack((np.zeros_like(sxbinary),sxbinary,s_binary))
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary ==1)] = 1
    #f,(ax1,ax2) = plt.subplots(1,2,figsize=(6,3))
    #ax1.set_title('stacked thresholds')
    #ax1.imshow(color_binary)
    #ax2.set_title('Combined S channel and gradient thresholds')
    #ax2.imshow(combined_binary, cmap='gray')
    return combined_binary



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


original = cv2.imread('test_images/test2.jpg')
org_dist = cv2.undistort(original,mtx,dist,None,mtx)
binary = pipeline(org_dist,sobel_kernel=3,sx_thresh=(20,100),s_thresh=(70,255))
#org_gray = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
#img_size = (1280,720)
img_size = (binary.shape[1],binary.shape[0])
binary_warped = cv2.warpPerspective(binary,M,img_size)
plt.imshow(binary_warped)
binary_warped.astype(int)

#Laneline pixels
def drawlanelines(original_image,binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)
    out_img = np.dstack((binary_warped,binary_warped,binary_warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:])+midpoint
    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0]-(window+1)*window_height
        win_y_high = binary_warped.shape[0]-window*window_height
        win_xleft_low = leftx_current-margin
        win_xleft_high = leftx_current+margin
        win_xright_low = rightx_current-margin
        win_xright_high = rightx_current+margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,
                  win_y_high),(0,255,0),2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,
                  win_y_high),(0,255,0),2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
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
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, 
                    binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*+ploty+right_fit[2]
    out_img[nonzeroy[left_lane_inds],nonzerox[left_lane_inds]] = [255,0,0]
    out_img[nonzeroy[right_lane_inds],nonzerox[right_lane_inds]]=[0,0,255]
    plt.imshow(out_img)
    plt.plot(left_fitx,ploty,color='yellow')
    plt.plot(right_fitx,ploty,color='yellow')
    plt.xlim(0,1280)
    plt.ylim(720,0)
    #Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    Minv = np.linalg.inv(M)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (original.shape[1], original.shape[0])) 
       
    #plt.imshow(result)
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
    #right_curverad.astype(int)
    # Now our radius of curvature is in meters
    #print(type(left_curverad))
    #print(left_curverad, 'm', right_curverad, 'm')
    
    # Combine the result with the original image
    cv2.putText(original_image, "left_curverad: %.3f m, right_curverad: %.3f m" % (left_curverad, right_curverad), 
                (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    return result

 
drawlanelines(original,binary_warped)
    
   
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
#plt.plot(histogram)
out_img = np.dstack((binary_warped,binary_warped,binary_warped))*255
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:])+midpoint
nwindows = 9
window_height = np.int(binary_warped.shape[0]/nwindows)
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
leftx_current = leftx_base
rightx_current = rightx_base
margin = 100
minpix = 50
left_lane_inds = []
right_lane_inds = []
for window in range(nwindows):
    win_y_low = binary_warped.shape[0]-(window+1)*window_height
    win_y_high = binary_warped.shape[0]-window*window_height
    win_xleft_low = leftx_current-margin
    win_xleft_high = leftx_current+margin
    win_xright_low = rightx_current-margin
    win_xright_high = rightx_current+margin
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,
                  win_y_high),(0,255,0),2)
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,
                  win_y_high),(0,255,0),2)
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
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
left_fit = np.polyfit(lefty,leftx,2)
right_fit = np.polyfit(righty,rightx,2)
ploty = np.linspace(0, binary_warped.shape[0]-1, 
                    binary_warped.shape[0])
left_fitx = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*+ploty+right_fit[2]
out_img[nonzeroy[left_lane_inds],nonzerox[left_lane_inds]] = [255,0,0]
out_img[nonzeroy[right_lane_inds],nonzerox[right_lane_inds]]=[0,0,255]
plt.imshow(out_img)
plt.plot(left_fitx,ploty,color='yellow')
plt.plot(right_fitx,ploty,color='yellow')
plt.xlim(0,1280)
plt.ylim(720,0)

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

out_img = np.dstack((binary_warped,binary_warped,binary_warped))*255
window_img = np.zeros_like(out_img)
out_img[nonzeroy[left_lane_inds],nonzerox[left_lane_inds]]=[255,0,0]
out_img[nonzeroy[right_lane_inds],nonzerox[right_lane_inds]]=[0,0,255]
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin,ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,ploty])))])
left_line_pts = np.hstack((left_line_window1,left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin,ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,ploty])))])
right_line_pts = np.hstack((right_line_window1,right_line_window2))
cv2.fillPoly(window_img,np.int_([left_line_pts]),(0,255,0))
cv2.fillPoly(window_img,np.int_([right_line_pts]),(0,255,0))
result = cv2.addWeighted(out_img,1,window_img,0.3,0)
plt.imshow(result)
plt.plot(left_fitx,ploty, color='yellow')
plt.plot(right_fitx,ploty,color='yellow')
plt.xlim(0,1280)
plt.ylim(720,0)


