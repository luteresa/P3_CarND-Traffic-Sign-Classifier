
[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image11]: ./examples/all_class_traffic_types.png "all_class_traffic_types"
[image12]: ./examples/classes_distribution.jpg "classes_distribution"
[image2]: ./examples/src_gray.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/train_accuracy.jpg "train_accuracy"
[image5]: ./examples/New_Images.jpg "New_Images"
[image6]: ./examples/predict_images.jpg "predict_images"
[image7]: ./examples/Top_proba_new_images.jpg "Top_proba_new_images"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

 * The size of training set, validtion set, and test set is:

    Training Set:   34799 samples

    Valid Set:     4410 samples

    Test Set:      12630 samples

 * The shape of a traffic sign image is 

    Image Shape: (32, 32, 3)

 * The number of unique classes/labels in the data set is 

    43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image11]

class contribution:

![alt text][image12]


## Design and Test a Model Architecture

### step1: Pre-process the Data Set

As a first step, I decided to convert the images to grayscale because the gray image works well in classification, and reduce the amount of calculation. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because normalized data is easier to converge in training.



In addition, i decided to generate additional data because the data is unbalanced.(It's not work well in current project, will be update in feature)

To add more data to the the data set, I used the following techniques 

•	Slight random rotations

•	Adding random shadows

Here is an example of an original image and an augmented image:

#![image][image3]

The difference between the original data set and the augmented data set is the following ... 

### step2:  Design model architecture

my final model architecture looks like below, consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x8 				|
| Dropout           | 0.5|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 13x13x8   |
| RELU             |                               |
| Max pooling        | 2x2 stride, outputs  6x6x26
| Fully connected		| 936x400        									|
| RELU             |                               |
| Fully connected     | 400x120                           |
| RELU             |                               |
| Fully connected     | 120x84                      |
| RELU             |                               |
| Fully connected     | 84x43                      |
| Softmax				| 43x1        									|

 


### step3: Train model. 

To train the model, I used an AdamOptimizer, and  hyperparameters shows below

EPOCHS = 30

BATCH_SIZE = 128

rate = 0.001

when epoch is bigger than 10, set learn rate to 0.0001.


My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.934
* test set accuracy of 0.935

validation set accuracy shows below:

![alt text][image4]

### step4: Test a Model on New Images

#### 1. download ten German traffic signs  on the web;

Here are five German traffic signs that I found on the web:

![alt text][image5]

The first image might be difficult to classify because the sample of "Speed limit(30km/h)" in X_train is too little.

#### 2. Here are the results of the prediction:

![alt text][image6]

Here are the results of the prediction:

| Image			        |     Prediction	        					|  result|
|:---------------------:|:---------------------------------------------:|:---------------:|
| Speed limit (30km/h)  | Speed limit (50km/h)   					| false |
| Pedestrians     	    | Pedestrians 								| true  |
| Turn right ahead      | Speed limit (30km/h)   					| false |
| Go straight or left   | Go straight or left 						| true  |
| Speed limit (60km/h)  | Speed limit (60km/h)   					| true  |
| Children crossing     | Children crossing 						| true  |
| Stop                  | Speed limit (30km/h)   				    | false |
| Yield     			| Yield 									| true  | 
| Turn right ahead      | Turn right ahead   						| true  |
| Wild animals crossing | Wild animals crossing 					| true  |


The model was able to correctly guess 7 of the 10 traffic signs, which gives an accuracy of 70%. 

This is  worse than the accuracy of the test set.

#### 3.  the softmax probabilities for each prediction. 

For the first image, the model is relatively sure that this is a Speed limit (50km/h) sign (probability of 0.99), but the image does contain a Speed limit (30km/h) sign. it's wrong.

The top five soft max probabilities were

![alt text][image7]

## problem: i think the project has problems is:

1.train set is too small and unbalance.






### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?




```python

```
