[**Traffic Sign Recognition**] 



---

[*Build a Traffic Sign Recognition Project*]

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[no entry]: "no_entry"

<img src="http://benchmark.ini.rub.de/Images/00020_00024.jpg" width="200" height="20" /> 

[no entry]: http://benchmark.ini.rub.de/Images/00020_00024.jpg "no_entry"



## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
[*Writeup / README*]

[*1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.*]

You're reading it! and here is a link to my [project code](https://github.com/byronrwth/Udacity-SelfDrivingCar-ND-Term1/blob/master/CNN/CarND-Traffic-Sign-Classifier-P2/P2.ipynb)

[*Data Set Summary & Exploration*]

[*1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.*]

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?  31319 images
* The size of the validation set is ?  3480
* The size of test set is ?  12630
* The shape of a traffic sign image is ?  32 * 32 pixel
* The number of unique classes/labels in the data set is ? 43 signs

[*2. Include an exploratory visualization of the dataset.*]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![a summary of training set](https://github.com/byronrwth/Udacity-SelfDrivingCar-ND-Term1/blob/master/CNN/CarND-Traffic-Sign-Classifier-P2/trainset.png)

[*Design and Test a Model Architecture*]

1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

    As a first step, I decided to convert the images to grayscale because color, brightness may differ according to various weather conditions, may not help much to identify signs;

    Optionally I considered to use localized histogram exposure to enhance low contrast images. But because validation accuracy has already reached 95%, so I leave exposure function not used.

    Here is an example of a traffic sign image before and after grayscaling.

![ color resized general caution ](https://github.com/byronrwth/Udacity-SelfDrivingCar-ND-Term1/blob/master/CNN/CarND-Traffic-Sign-Classifier-P2/color_caution.png)
![ gray resized general caution ](https://github.com/byronrwth/Udacity-SelfDrivingCar-ND-Term1/blob/master/CNN/CarND-Traffic-Sign-Classifier-P2/gray_caution.png)

    As a last step, I normalized pixel values from [0, 255] into [0, 1] for better matching activation function ranges, by dividing 255.0 ;

    I can rotate signs into 90 degree, 180 degree, 270 degree to generate more training data, because in this way I can makeup equal number of samples for each class type, therefore balance features trainings.



2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

    My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Fully connected		| Input = 400. Output = 120.        		    |
| RELU				    | SAME padding        							|
| Fully connected       | Input = 120 Output = 84                       |
| RELU                  | SAME padding                                  |
| Fully connected       | Input = 84 Output = 43                        |
                           
 


3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

    To train the model, I used cross validation of training set and validation set. I also shuffle train set each time when I start to train the parameters, in this way to avoid any fixed pattern in training data. 


4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

    I sued Adamoptimizer which is more efficient than normal gradient descent to find the optimum by dynamically changing the gradient. 

    the learning rate cannot be too large, after testing from 0.1 to 0.001, I set to be .001;
    Batch size should between 100 to 512, I set 128.

    Dropout is necessary but for simple architecture, don't set too small. I change from 0.9 to 0.8

    I ran it for 30 epochs, after 16 epoch the validation accuracy reaches 95% and stopped.

    Weights set to the default values of 0 mean and .1 standard deviation to start to resemble a normal distribution.

## My final model results were:

[*validation set accuracy of*] > 95%
[*test set accuracy of*] 88%


## If an iterative approach was chosen:

[*What was the first architecture that was tried and why was it chosen?*]  
I used classical LeNet architecture

[*What were some problems with the initial architecture?*]
this LeNet is initially used to identify hand written digits, therefore it is relatively weak to match more complex features from traffic signs, it is better to use more feature filters in 1st and 2nd convolution layers

[*How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.*]
without using dropout it will get overfitting, also the size of batch for back propagation will effect in efficiency to optimize the loss function


[*Which parameters were tuned? How were they adjusted and why?*]
set dropout to be 0.8, batch size = 128, this will make optimization process converged and slowly test accuracy goes up to 88%

[*What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?*]
dropout, max-pooling will help


## If a well known architecture was chosen:

[*What architecture was chosen?*]
LeNet architecture which has 2 convolution layers and 3 fully-connected layers


[*Why did you believe it would be relevant to the traffic sign application?*]
LeNet is aimed to identify classes which differs in shapes, not colors


[*How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?*]
after 20-30 epochs, my test accuracy reaches 88%, validation reaches 95%
 

## Test a Model on New Images

[*1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.*]

Here are five German traffic signs that I found on the web:

![ 30 zone ](https://www.google.de/imgres?imgurl=http%3A%2F%2Fcdn.xl.thumbs.canstockphoto.com%2Fcanstock14957677.jpg&imgrefurl=http%3A%2F%2Fwww.canstockphoto.com%2Fimages-photos%2Ftraffic-signs-maximum-speed.html&docid=-DhzUuddTPMhxM&tbnid=jBCnFxElHvgl2M%3A&vet=10ahUKEwjqkMmhvqnTAhWjJcAKHVNRDgUQMwhjKC0wLQ..i&w=270&h=194&safe=off&bih=758&biw=1720&q=german%20traffic%20signs&ved=0ahUKEwjqkMmhvqnTAhWjJcAKHVNRDgUQMwhjKC0wLQ&iact=mrc&uact=8)
![ general caution sign after human's face ](http://storage.torontosun.com/v1/blogs-prod-photos/5/0/d/a/e/50dae8694fd60fb29cec2d767c897bb0.jpg?stmp=1290376611)
![ priority for oncoming vehicle ](https://www.google.de/imgres?imgurl=https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2F9%2F91%2FZeichen_208.svg%2F120px-Zeichen_208.svg.png&imgrefurl=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FRoad_signs_in_Germany&docid=LsOI0ct_PJpQ9M&tbnid=HXmtpvAESe0INM%3A&vet=10ahUKEwjqkMmhvqnTAhWjJcAKHVNRDgUQMwhXKCEwIQ..i&w=120&h=120&safe=off&bih=758&biw=1720&q=german%20traffic%20signs&ved=0ahUKEwjqkMmhvqnTAhWjJcAKHVNRDgUQMwhXKCEwIQ&iact=mrc&uact=8)
![ road work sign under sunshine ](https://www.google.de/imgres?imgurl=http%3A%2F%2Fa.rgbimg.com%2Fcache1nHmS6%2Fusers%2Fs%2Fsu%2Fsundstrom%2F300%2FmifuUb0.jpg&imgrefurl=http%3A%2F%2Fwww.rgbstock.com%2Fphoto%2FmifuUb0%2FTraffic%2BSign&docid=lOWu3Z0lDLKqrM&tbnid=6gOYLwt7br8W6M%3A&vet=10ahUKEwjqkMmhvqnTAhWjJcAKHVNRDgUQMwhbKCUwJQ..i&w=300&h=225&safe=off&bih=758&biw=1720&q=german%20traffic%20signs&ved=0ahUKEwjqkMmhvqnTAhWjJcAKHVNRDgUQMwhbKCUwJQ&iact=mrc&uact=8)
![ yellow diamond in village ](https://www.google.de/imgres?imgurl=http%3A%2F%2Fbicyclegermany.com%2FImages%2FLaws%2FArterial.jpg&imgrefurl=http%3A%2F%2Fbicyclegermany.com%2Fgerman_bicycle_laws.html&docid=hhnc9xU3HHVSrM&tbnid=uGg7Wbl1OO8DqM%3A&vet=10ahUKEwjqkMmhvqnTAhWjJcAKHVNRDgUQMwhaKCQwJA..i&w=640&h=480&safe=off&bih=758&biw=1720&q=german%20traffic%20signs&ved=0ahUKEwjqkMmhvqnTAhWjJcAKHVNRDgUQMwhaKCQwJA&iact=mrc&uact=8)


The image "30 zone" might be difficult to classify because orginal image will be resized to 32,32 ,which may lost pixels, also the sign itself is Only 30% large in the whole picture.

The image "general caution sign after a human face" might be difficult to classify because a human's face takes 50% of whole and in front of sign.

The image "priority for oncoming vehicle" might be difficult to classify because the original red arrow indicates you have lower priority, but in gray red turns into brighter gray.


The image "road work sign under sunshine" might be difficult to classify because good weather, too bright sunshine.

The image "yellow diamond in village" might be difficult to classify because yellow diamond may or may not be covered in class type.



[*2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).*]

Here are the results of the prediction:

| Image			                    |     Prediction	        					    | 
|:---------------------:|:---------------------------------------------:| 
| 30 limit zone      	            | road work     									| 
| general caution      	            |  road work 										|
| priority for oncoming vehicle		| Speed limit (60km/h)								|
| road work	      		            | Wild animals crossing					 			|
| main road		                    | Priority road      							    |


The model was able to correctly guess 1 of 5

[*3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)*]

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the "yellow diamond in village" image, the model is relatively sure that this is a stop sign (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Priority road   								| 
| .001     				| Yield 										|
| .001					| Speed limit (50km/h)							|
| .001	      			| No vehicles					 				|
| .001				    | Keep right     							    |


For the "30 zone" image, the model is relatively sure that this is a Road work sign (probability of 0.99). 

For the "general caution sign after a human face" image, the model is relatively sure that this is a Turn right ahead sign (probability of 0.85). 

For the "priority for oncoming vehicle" image, the model cannot tell whether it is Road work sign (probability of 0.46), or Priority road (8%), Keep right (8%), General caution(8%)

For the "road work sign under sunshine" image, the model is relatively sure that this is a Wild animals crossing sign (probability of 0.91). 



## (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
[*1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?*]

in layer 1 convolution, which has 6 filters of 5*5, so I can see 6 features.  maybe becuase I reused the LeNet for hand written identify,  these features seem to try to locate in which area your symbol locates, in middle bottom, or left, or central, or middle up ?


in layer 2 convolution, which has 16 filters of 5*5, so I can see 16 features, which focus on finding some edge patterns of symbols ?


