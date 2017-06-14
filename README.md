# Udacity-SelfDrivingCar-ND-Term1
following udacity SDC nanodegree Term1 projects and learning materials

Computer Vision:
- P1: develope a pipeline in python for lane detection, used OpenCV libary to implement Canny Edge Detection and Hough transform to detect line segments of lanes

- P4:  compute the camera calibration, apply distortion correction to raw images, and then use color transforms, gradients, etc., to create thresholded binary images and apply perspective transform on them to detect lane pixels; determine the curvature of the lane by numerical estimation and output visual display of the lanes

- P5: perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images clipped from video stream, and train a classifier Linear SVM classifier with these images; implement a sliding-window technique and use your trained classifier to search for vehicles in images, create a heat map of recurring detections frame by frame to reject outliers and visually display the estimated a bounding box for vehicles detected

Deep Learning:
- P2:  use online German Traffic Sign Dataset to train my CNN architecure which bases on LeNet Model plus dropout, max-pooling layers;  the model is implemented with Keras and Tensorflow;  parameters are  tuned with Adam Optimizer, and after 20-30 epochs,  test accuracy reaches 88%, validation reaches 95% 

- P3: use Udacity simulator to drive the vehicle and collect the train data of good driving behavior, including center, left and right camera images; pre-process these data by cropping out non-road image part and dropping some data out of training set to guarantee distribution over each steer angle equally; implement my CNN model in Keras and Tensorflow, which bases on Nvidia model plus dropout; use batch generator to guarantee memory usage and train on AWS to reduce time. Finally in simulator autonomous mode the vehicle can farward and stay on the lane.

