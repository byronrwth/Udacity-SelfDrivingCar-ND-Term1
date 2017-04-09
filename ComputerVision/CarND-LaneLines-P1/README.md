[**Finding Lane Lines on the Road**]



---

[*Project Specification*]

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report
* Include all project files with submission


[//]: # (Image References)

![grayscale]<img src="https://github.com/udacity/CarND-LaneLines-P1/blob/master/examples/grayscale.jpg" width="200" height="200" />
![line-segments-example]<img src="https://github.com/udacity/CarND-LaneLines-P1/blob/master/examples/line-segments-example.jpg" width="200" height="200" />


---

[*Reflection*]


[*1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.*]

My pipeline consisted of 8 steps. First, I converted the images to grayscale, then I define a kernel size of 5 for Gaussian smoothing. After this, I define low_threshold of 100, high_threshold of 200 for Canny Edge Detection. My triangle region-of-interest is [(W-W/20,H),(W/2,H/2),(W/20,H)], image pixels which are outside of this triangle area will be ignored for further detections. Then I use cv2.fillPoly to fill pixels inside the triangle area with the fill color "black", thus get a masked image. From the masked image, I set rho=1, theta=np.pi/180, threshold=40, min_line_len=10, max_line_gap=20 for Hough transform and get an array of line segments. Next, I need to draw left and right lanes for these points.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by looping all segments and calculating each slope and bias of each segment, if slope < 0 and slope > -infinate, store this negative slope and bias into right lane group, so does left lane group. Then for each group, I average all slopes and all bias, assuming in this way I can draw 2 single lane, to across right and left group of segments accordingly.

If you'd like to include images to show how the pipeline works, here is how to include an image:

![founded lanes marked with white lines](https://github.com/byronrwth/Udacity-SelfDrivingCar-ND-Term1/blob/master/ComputerVision/CarND-LaneLines-P1/found_lanes.png)

[challenge video](https://youtu.be/isCo9Dj7DhA)



[*2. Identify potential shortcomings with your current pipeline*]


One potential shortcoming would be what would happen when there are multiple lanes exist in the triangle region of interests. This would lead to multiple right and left lanes to be detected at the same time. And on very sharp curve roads, it would also be difficult to calculate good slopes for right and left lanes.

Another shortcoming could be in bad road conditions, there are no obvious lane segments to be detected; or within road construction area, there are temporary yellow lanes(which suppose to be followed) and original white lanes are crossing each other.


[*3. Suggest possible improvements to your pipeline*]

A possible improvement would be to select the highest positive slope for left lane, and select the smallest negative slope for right lane, i.e. find the narrowest left lane, right lane group as your car's lanes.

Another potential improvement could be to ...
