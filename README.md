# SafeWalk
Title:
SafeWalk
Yangming Chong (yangminc) Jiayi Lin (jiayilin)

Youtube link:
https://www.youtube.com/watch?v=vcMwC26Ra6I&feature=youtu.be

Summary:                  
In this project we propose a framework to use the back camera on a mobile device to detect obstacles on the ground and alert the visually impaired user. We use statistical method to compare the newly captured scene with the floor in a previous step. The intensity of grayscale pixels, the hue of the objects and the distribution of ORB keypoints are explored and used as statistics. An edge detector is also used to increase the robustness of the framework in highly similar environments. The accuracy, running time and robustness of the framework are tested and discussed. We managed to achieve a high frame rate as the algorithm has to run in real time when the user is walking at a certain speed. A learning based method is also implemented as a baseline to compare with our framework.  

Background:                  
The visually impaired people have to use their guide canes to detect obstacles when they are walking on the street or inside buildings. It’s surprising that there’s no available mobile phone applications to assist the visually impaired people to avoid obstacle. With the resolution of cell phone cameras being higher and the mobile CPUs being more powerful, a mobile safe walking assistance solution becomes possible. 
The user just needs to hold the cell phone at a comfortable height in front him/her and slightly tilt the phone up so that the camera can view the ground within several meters in front of the user. The app keeps reading the frames from the camera at a relatively low frame rate (less than 2Hz should suffice). These frames are used to calculate the statistics of the ground pattern. When the pattern changes in a new image, an obstacle is reckoned to be spotted.
Our application involves utilizing the OpenCV framework for iOS, accessing the camera and reading from the IMU. OpenCV provides us easy-to-use feature detector and extractors such as ORB, as well as classifiers such as SVM. We will build these algorithms into a pipeline to realize the full functions. When estimating the distance from the obstacle, the IMU data can be useful because it tells us the walking speed of the user and the height of the cell phone. 

Challenge:                  
Obstacle detection using a monocular camera is a difficult task in real daily settings due to the highly diverse appearance of the ground and objects. Different ground patterns include tiles, concrete, marble, carpet (numerous patterns inside this), wooden floor, etc. Obstacles include pedestrians, vehicles, still objects, walls, stairs, gaps or holes, etc. Thus, a robust algorithm is needed to deal with most, if not all, of these cases.
Another challenge lies in the speed of the mobile application. Assume the person walks at the speed of 0.5 m/s, the camera sees 2 meters in front of the user, and the user needs to be alerted more than 1 meter away from the obstacle to avoid danger. Then the algorithm has to be able to detect the obstacle in less than 2 seconds. In real situations, due to the blur of the moving camera and the size of the obstacle, a running time less than 0.5 second is needed for good user experience. 

Thus, we did a large number of tests in various environments and develop a four-layer framework as shown in the figure below to address the challenges discussed above. They are non-learning based methods. This saves us the complexity of training a model and also makes each of the four layers extremely fast. They are executed only if certain conditions are met, so that the actual running time is very little.
The contribution of the project does not lie in any novel mathematical algorithm or hardware acceleration technique. Instead, we developed a fast obstacle detection framework that leverages some modern algorithms, which can solve a practical problem. We hope our method of combining several separate algorithms and optimize them for a real-world problem could inspire other people to develop better algorithms or applications to facilitate people’s daily life.

![alt tag](https://github.com/jiayilin/SafeWalk/blob/master/structure.png)


Goals & Deliverables:                                    
Plan to achieve:                   
a. Finish the pipeline shown in the flow chart without the offline classifier. 
b. Able to work in real time
c. Recall rate > 80%
d. Easy-to-use user interface

Hope to achieve:                                     
a. offline classifier
b. Precision rate > 60%
c. Build depth image from multiple images and classify based on position of the points
Evaluation:
Use the app in the real world and check if a. it detects obstacles with the required recall rate, b. it works in real time without delay, c. anyone can quickly learn to use the app and feel comfortable using it.

Schedule:                  
Nov 09 - Nov 15                   
Yangming Chong: Generate ORB features from the images                   
Jiayi Lin: Load consecutive images from the camera

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
Jiayi Lin: Create a user-friendly interface 

Dec 7 to Dec 10                   
Yangming Chong: Make the video; Write the report                   
Jiayi Lin: Make the video; Make the presentation


