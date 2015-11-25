# SafeWalk
Title:
SafeWalk
Yangming Chong (yangminc) Jiayi Lin (jiayilin)

Summary:
In this project we plan to use the front camera to capture the images of the ground and send notification to alert the user if there are obstacles on the way. We plan to use statistical method to detect the feature of the ground and compare the current feature with the statistic result. For feature detection, we plan on employing ORB from OpenCV to speed up the process to run in real time. For classification, we want to explore online SVM to update the feature dictionary incrementally and classify the result based on all the statistic information. We want to provide alert to the user in two seconds if an obstacle is detected to make sure the user can walk safely.  

Background
The visually impaired people have to use their guide canes to detect obstacles when they are walking on the street or inside buildings. It’s surprising that there’s no available mobile phone applications to assist the visually impaired people to avoid obstacle. With the resolution of cell phone cameras being higher and the mobile CPUs being more powerful, a mobile safe walking assistance solution becomes possible. 
The user just needs to hold the cell phone at a comfortable height in front him/her and slightly tilt the phone up so that the camera can view the ground within several meters in front of the user. The app keeps reading the frames from the camera at a relatively low frame rate (less than 2Hz should suffice). These frames are used to calculate the statistics of the ground pattern. When the pattern changes in a new image, an obstacle is reckoned to be spotted.
Our application involves utilizing the OpenCV framework for iOS, accessing the camera and reading from the IMU. OpenCV provides us easy-to-use feature detector and extractors such as ORB, as well as classifiers such as SVM. We will build these algorithms into a pipeline to realize the full functions. When estimating the distance from the obstacle, the IMU data can be useful because it tells us the walking speed of the user and the height of the cell phone. 

Challenge
The challenges of the project lie in the speed of training the classifier, the accuracy and the robustness of the feature detection.
We plan to use SVM implementation of OpenCV (based on LibSVM) to build the classifier. Since keeping bringing in new image features and deleting old image features to maintain a certain number of images for classification can be time consuming, we come up with two approaches to speed up the process. The first one we look into is to build an online SVM classifier that trains the data incrementally. So we do not need to retrain the whole data and can only update the classifier by adding the result of the new feature.  Another approach is, instead of doing all the training online, to gather large number of the images and train them offline. So we have an offline classifier and whenever a new image come in, it can quickly give the result based on all the offline training data. We hope that by trying these two approaches, we can speed up the classification process, to quickly classify the result given a new image and respond to the user as soon as possible.
The second challenge lies in feature detection. Since pictures are taken by the user, the pictures may be taken from different angle and the lighting environment can also change a lot from pictures to pictures. As the texture of the ground can be different from the carpet, brick to outside road, there may also be problem that when the user move from one texture of the ground to the other, the feature changes dramatically and can be classified as obstacles.  

![alt tag](https://github.com/jiayilin/SafeWalk/blob/master/flowChart.png)

Note: The “Online classifier” in the flow chart is different from the “online SVM” mentioned earlier. Here the online classifier is the opposite of the offline classifier. The online classifier keeps being retrained using the images captured by the user on-the-fly while the offline version is pre-trained and doesn’t get updated during the walk. The online SVM is a different version from the standard SVM, which keeps adding new single samples and update the weights instead of batch training. Both online SVM and standard SVM are online classifiers.

Goals & Deliverables
Plan to achieve: a. Finish the pipeline shown in the flow chart without the offline classifier. 
b. Able to work in real time
c. Recall rate > 80%
d. Easy-to-use user interface

Hope to achieve: a. offline classifier
b. Precision rate > 60%
c. Build depth image from multiple images and classify based on position of the points
Evaluation:
Use the app in the real world and check if a. it detects obstacles with the required recall rate, b. it works in real time without delay, c. anyone can quickly learn to use the app and feel comfortable using it.

Schedule                                             
Nov 09 - Nov 15  
Yangming Chong: Generate ORB features from the images 
Jiayi Lin:      Load consecutive images from the camera
Nov 16 - Nov 22                   
(Collaboratively) Select and train the bag of words classifier
Nov 23 - Nov 25                   
(Collaboratively) Improve the speed and detection accuracy by using greyscale MRSE 
Nov 25 to Nov 28
Yangming Chong: Add the color channel to improve the accuracy of the algorithm
Jiayi Lin: Connect the camera, add vibration for notification, and test it on the ipad
Nov 29 to Dec 2
Yangming Chong: Add filter bank to train the images 
Jiayi Lin: Collect more training data and create a new dictionary
Dec 3 to Dec 6
Yangming Chong: Adjust the algorithm to be turning-safe
Jiayi Lin: Create a user-friendly interface with voice notification
Dec 7 to Dec 10
Yangming Chong: Make the video; Write the report
Jiayi Lin: Make the video; Make the presentation


