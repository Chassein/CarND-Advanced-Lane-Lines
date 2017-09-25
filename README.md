
# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In the following, I describe my solution to the Advanced Lane Finding Project of the Udacity - Self-Driving Car NanoDegree. I will consider the [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points individually and describe how I implemented them.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: distorted_vs_undistored_chess.PNG "Distorted vs Undistorted - Chess"
[image2]: distorted_vs_undistored.PNG "Distorted vs Undistorted"
[image3]: normal_vs_warped.PNG "Warp Example"
[image4]: all_color_channels.PNG "All color channels"
[image5]: color_channels_combined.PNG "Combining color channels"
[image6]: fitting_procedure.PNG "Fitting Lane Lines"
[image7]: curvature.PNG "Curvature of left and right Lane Line"
[image8]: offset_from_center.PNG "Offset from the Center of the Lane"
[image9]: final_lane.PNG "Fitted Lane"

---

## Camera Calibration

I implemented the camera calibration as proposed in the `examples.ipynb`. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration matrix and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function. The result looks as follows (left the distorted image, right the undistorted image):

![alt text][image1]

---

## Pipeline

### 1 Preprocessing
The following two preprocessing steps are applied to each image at the beginning of my pipeline.

#### 1.1 Distortion-correction

First, I used the camera calibration matrix and distortion coefficients computed by the camera calibration step to remove the distortions from the image. The result looks as follows (left the distorted image, right the undistorted image):

![alt text][image2]


#### 1.2 Perspective transform

First, I selected a mask which marks the region of the video which is most important to find the lane lines.
Next, I specified destination points such that lines which were straight and parallel in the real world appear straigth and parallel in the warped image. I use the following coordinates for the transformation:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 65, 720       | 205, 720      | 
| 760, 460      | 1280, 720     |
| 520, 460      | 0, 0          |
| 1215, 720     | 1075, 720     |

To compute the transformation matrix (and the corresponding inverse) is used the `cv2.getPerspectiveTransform()` function. This matrix is then `warp_image()` function, which uses the `cv2.warpPerspective()`

The result of the perspective transform looks as follows (left the original image, right the transformed image):

![alt text][image3]

### 2 Picking good color channels

To find meaningfull color channles, I wrote a pipeline which applies the two described preprocessing steps to each image. Further, it computes the three R,G,B and the three H,L,S color channles. For each color channel the x and the x+y gradients are computed. This results in 18 additional frames for one original frames. Stiching everything together produces the following result.

|     | Column 1      | Column 2      |  Column 3     | Column 4      |  Column 5     | Column 6      | 
|:---:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:| 
|**Row 1**|Original|Original|Original|B|Grad_x B| Grad_xy B|
|**Row 2**|Original|Original|Original|H|Grad_x H| Grad_xy H|
|**Row 3**|R|Grad_x R|Grad_xy R|L|Grad_x L| Grad_xy L|
|**Row 4**|G|Grad_x G|Grad_xy G|S|Grad_x S| Grad_xy S|

![alt text][image4]


The S channel usually provides either very strong or rather weak responses to lane lines. On the other hand, the R channel is also reasonable to identify white and yellow lane lines. It is a bit more consistent than the S channel. To get the best of both channels I used the maximum operation to combine them. The result looks as follows (on the left the R channel, on the middle the S channel, and on the right the maximum of both):

![alt text][image5]


### 3 Which point belongs to a lane line?

After defining my new color space, I had to decide tresholds to indicate which point belongs to a lane line. I used the following rule to find good points:

A pixel is chosen if the value of this pixel is above **190** or is the value of the pixel is above **170** and the absolute value of its x gradient (kernel size of 5) is at least **330**.

Note: The x gradient alone is a rather bad indicator for lane lines, since it also marks the transition from a dark to a ''not so dark'' region. Which is usally not a good idea if one searches for lane lines. 

### 4 Fitting lane lines through the chosen points

For the first frame of the video, I used the function provided by the Udactiy lecture, which uses the sliding window technique. For all other frames, I used my own pipeline. The main idea of my pipeline is to remember the  lane lines found at the last frame. For the acutal frame, I consider candidate points, only if they lie inside a corridor which is defined by the lane line found on the last frame. In this way, noise which is not near the previous lane line is ignored. 

All points which fulfill the above mentioned rule and which lie inside a corridor are then used to fit a polynomial. First, I fit a line through the points and compute sum of residuals using the function `np.polyfit()`. Next, I fit a 2nd order polynomial through these points using the same function. Only if the sum of residuals for the 2nd order polynomial is below **75%** of the sum of residuals for the line, I use the 2nd order polynomial. Otherwise, I stick with the line. In this way, I reduce the problem of overfitting the candidate points.

Note: If less than **40** points are found, I don't trust these points, instead I just keep the lane line found from the last frame. Further, I don't use directly the new found lane line, instead I use a **50:50** average from the lane line from the last frame and the lane line from this frame, by averaging the coefficients of the polynomial.

The following picture visualizes the fitting procedure. The picture contains three color channels. The red color channel indicates the value of each point (maximum of R and S channel). The green color channel indicates if a point fulfills the criteria to be chosen. If the point is chosen, its green value is set to **255**. The blue channel marks the corridor defined by the previous lane lines. If a point is inside this corridor, its blue value is set to **255**.

![alt text][image6]

First, note that no points form the black line are chosen, beside their high high gradients. Further, the left line is not really bright, however the boundaries of the line fulfill the second requirement of the rule and are therefore selected. Second, other bright points which may distract from the true lane lines are masked out by using the blue corridors.


### 5 Computing the radius of curvature of the lane and the position of the vehicle with respect to center

During the video I compute the lane curvature from the coefficients which describe the left and right lane lines. I used the formulas and parameters (which are necessary to transform the values from pixel space to real world) as proposed in the Udactiy lecture notes. To compute the offset, I evaluate the two parabels describing the lane lines at the bottom of the picture and average the corresponding values. To obtained the offset (in pixel space) I substracted the obtained value from the middle of the picture. Last, I convert the pixel value to m using the proposed parameters.

To draw the computed values onto each frame of the video I used the PIL library. Since the lane lines tend to be straight (which results in an infinite curvature radius) I clipped the computed values at 9999 m.

### 6 Drawing the Lane

To draw the lane area in the original frame I first compute two set of points which represent the two lane lines. Second, I use the function `cv2.fillPoly()` to mark the region between these two sets. To map these region on the original image I use the inverse of the transformation matrix as an input to the function `cv2.warpPerspective()`. The so obtained image is added to the original image with `cv2.addWeighted()`. The following picture presents an example of the final result.

![alt text][image9]

---

# Final Videos

The described pipeline is very sucessfull in identifiny the lane lines for `project video.mp4` and `challenge_video.mp4`. See  [solution_project.mp4](solution_project.mp4) and  [solution_challenge.mp4](solution_challenge.mp4) for the final results. However, on the `harder_challenge_video.mp4` the pipeline fails. I briefly describe in the Bonus section, the vastly more complex pipeline which I used to tackle this video. The results are not perfect but still quite satisfying: [solution_harder_challenge.mp4](solution_harder_challenge.mp4)

---

# Bonus

The first two videos are recorded in rather good circumstances - compared to the third video. I decided to build a new specialized pipeline to tackle this video. The following are major differences between the first two and the third video:

* The road is really curvy, hence the visible part of the lane line is significantly shorter
* The road is partially very bright and sometimes very dark
* At some parts of the road only the left or right lane line is clearly visible
* At one point the lane line is not even visible in the video

To adresse these points I did the following adaptions to my pipeline

** The road is really curvy, hence the visible part of the lane line is significantly shorter **

* Instead of taking the complete corridor of the last line, I took only the lower half of the corridor
* I increased the belive of the new lane line to avoid for quicker adaptions to the curvy road

** The road is partially very bright and sometimes very dark **

* Instead of using color and gradient information over the complete corridor, I split the corridor in smaller sections. In each section, I first try to find enough bright points, if this fails I use additional gradient information. In this way, I can cope with siutations where parts of the lane line are hidden under shadows
* If one part of the corridor is filled almost completly by bright points, this has nothing to do with the appareance of a lane line (the image is just very bright at this position). Therefore, I decided to remove the bright points of this section to make the fitting procedure not over excited about the bright part

** At some parts of the road only the left or right lane line is clearly visible **

* To decide how certain the model is about position of a lane line, I developed a score function for each lane line. I use exponential smoothing to track the score of each lane line.
* It may happen that one of the two lane lines, is not correctly identified. Therefore, I developed a function to check if the two lane lines are compatible (describing a meaningfull road). If the lines are not compatible, I evaluate different plausible scenarios and choose the best one (evaluated with respect to the previously mentioned score function).

**  At one point the lane line is not even visible in the video **
* To handle this special situation, I checked if too few points are describing one line, in this case I used the other line to compute the line.

In summary, many (very video specific) functions and parameter choices were made, which produce a quite reasonable solution for the third video. However, the found lane lines are still not perfect.

---

## Discussion

The approach to directly find a binary threshold video from the different color channels, which is meaningfull for the lane finding task, is rather hard. Integrating this step into the lane finding process, by using corridors (or cells in these corridors) helps. However, the approach is still limited. Many parameters needs to be chosen to get good results. The more parameters are set the lower is the chance to produce a pipeline which generalizes well. 

To improve the pipeline I propose to use a small convolutional neural network which takes a small part of the input image and decides if this part contains a lane line. Generating training data may either be done by randomly generating some image pages with lane lines and without lane lines or by manually labeling some parts of the input. As learned, a convolutional neural network is very powerfull feature extractor of images and should therefore be perfectly for this task. The output of the neural network should then be used as an input for the lane finding problem and produce quite accurate results.
