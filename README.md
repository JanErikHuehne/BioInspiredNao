# BioInspiredNao
This repository contains the source code for the project Biologically Inspired Learning for Humanoid Robots 

## Technology
The project is based on the [NAO robot platform](https://www.aldebaran.com/en/nao). Communication with the robot is based 
on ROS. All implementation are done in Python2 (due to limited supporting plaform of ROS). 

## Red Object Tracking and Mirroring Arm Movement
The implementations of this task has been done in multiple steps:
1. The red object tracking is operformed using OpenCV. After preprocessing and detection of red blobs in the input image the center of the largest
 red blob of the image is saved
2. We use the the front head tactile touch button to collect training samples. When touched a new training sample containing the current position of the biggest
   red blob as well as the current states of the shoulder joints (roll and pitch) is added to a list of training samples.  
3. We implemented a multi-layer neuronal network from scratch using only numpy. Using ADAM as an optimizer, we trained this network with the collected training samples.
4. Afterwards, the behaviour of the trained algorithm could be demonstrated

![Alt Text](ezgif-5-9c296fb4f0.gif)
