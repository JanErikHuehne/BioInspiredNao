#!/usr/bin/env python

"""
Group:

    Jan-Erik Huehne
    Oscar Soto
    Joseph Gonzalez
"""

import rospy
from argparse import ArgumentParser
from naoqi import ALProxy
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from tutorial_3.scripts.nao_nn import *
import pickle 


class Central:


    def __init__(self, train_mode=False, reset_mode=False):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0
        self.stiffness = False  
        self.repetitive_flag = 0
        self.repetitive_state = 0
        self.mirrored = 0
        self.last_mask = None
        self.center = None
        self.proxy = None
        # If this parameter is set, stiffness will be set to allow the collection of training samples
        self.train_mode = train_mode
        self.reset = reset_mode
        # We import the trained NN 
        with open("nn.pkl", "rb") as f:
            self.neuralnet = pickle.load(f)
        pass


    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def joints_cb(self,data):
        # store current joint information in class variables
        self.joint_names = data.name 
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        pass

    def bumper_cb(self,data):
        rospy.loginfo("bumper: "+str(data.bumper)+" state: "+str(data.state))
        if data.bumper == 0:
            self.stiffness = True
        elif data.bumper == 1:
            self.stiffness = False
        # rospy.loginfo(get_touch_state())
            

    def touch_cb(self,data):
        # tactile buttons cases
        #   button 1: If the training mode is enables we will use the collected sample of shoulder joint angles and red blob position to add a new training sample to our dataset
        if data.button==1 and data.state==1:
            
            if self.train_mode:
                with open("data2.txt","a") as f:
                    f.write("{} {} {} {} \n".format(self.center[0], self.center[1], self.proxy.getAngles("LShoulderPitch",True)[0], self.proxy.getAngles("LShoulderRoll",True)[0] ))
        
               

            


    def image_cb(self,data):
        bridge_instance = CvBridge()
        try:
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")
            #print(cv_image.shape)
            frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            # two masks to capture red color in HSV
            # lower boundary RED color range values; Hue (0 - 5)
            lower1 = np.array([0, 100, 20])
            upper1 = np.array([5, 255, 255])
        
            # upper boundary RED color range values; Hue (175 - 180)
            lower2 = np.array([175,100,20])
            upper2 = np.array([180,255,255])
            # create the binary masks    
            lower_mask = cv2.inRange(frame, lower1, upper1)
            upper_mask = cv2.inRange(frame, lower2, upper2)
        
            full_mask = lower_mask + upper_mask
            # apply morphological opening to remove noise
            full_mask = cv2.erode(full_mask,None,iterations = 1)
            full_mask = cv2.dilate(full_mask,None,iterations = 1)
            # noise reduction by averaging with the last time point frame
            if self.last_mask != None:
                full_mask, self.last_mask = cv2.bitwise_and(full_mask, self.last_mask), full_mask
            # extract contours of every blob
            _, contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # if at least one blob 
            if contours:
                # get largest blob center ponint and log it 
                blob = max(contours, key=lambda el: cv2.contourArea(el))
                M = cv2.moments(blob)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                self.center = center
                #rospy.loginfo(center)
                # print(center)
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            cv2.imshow('frame',frame)

            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice


        except CvBridgeError as e:
            rospy.logerr(e)
        
        cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly

    # sets the stiffness for all joints. can be refined to only toggle single joints, set values between [0,1] etc
    def set_stiffness(self,value):
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
        try:
            stiffness_service = rospy.ServiceProxy(service_name,Empty)
            stiffness_service()
        except rospy.ServiceException as e:
            rospy.logerr(e)

    def set_joint_angles(self,head_angle,joint_name,relative=False, speed=0.1):

        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(joint_name) # each joint has a specific name, look into the joint_state topic or google
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = relative # if true you can increment positions
        joint_angles_to_set.speed = speed # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)
        
    def setup_for_training(self):  
        ip_nao = "10.152.246.54"
        port = 9559
        try:
            self.proxy = ALProxy("ALMotion",ip_nao,port)
        except Exception as e:
            print("Couldnt create proxy to ALMotion")
            exit(1)
        
        # we use the proxy to set the individual stiffness of the joints
        names_stiffness = ["Head","LElbowRoll","LElbowYaw","LShoulderPitch", "LShoulderRoll"]
        value_stiffness = [0.9,0.9,0.9,0.0,0.0]

        for i in range(len(names_stiffness)):
            self.proxy.stiffnessInterpolation(names_stiffness[i], value_stiffness[i],1.0)

    def central_execute(self):
        
        
       
        # rospy.loginfo("asda")
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)

        # # test sequence to demonstrate setting joint angles

        rate = rospy.Rate(10) # sets the sleep time to 10ms
        if self.train_mode:
                self.setup_for_training()
        elif self.reset:
            self.set_stiffness(False)
        else: 
            self.set_stiffness(True)
        while not rospy.is_shutdown():
            
            
            if self.center != None and not self.train_mode:
                # We use the trained the neural network to get the shoulder joint angles predictions
                prediction = self.neuralnet.predict(np.reshape(np.array(self.center)/255.0,newshape=(2,1)))
                # We de-normalize the values 
                #dn_prediction =  (prediction/10)-1
                pitch_pred = (prediction[0,0] * (self.neuralnet.pitch_max_norm-self.neuralnet.pitch_min_norm)) + self.neuralnet.pitch_min_norm
                roll_pred = (prediction[1,0] * (self.neuralnet.roll_max_norm-self.neuralnet.roll_min_norm)) + self.neuralnet.roll_min_norm
                # Set the respective joint angles 
                print(pitch_pred, roll_pred)
                
                self.set_joint_angles(pitch_pred, joint_name="LShoulderPitch")
                #self.set_joint_angles(1.0, joint_name="LShoulderPitch")
                self.set_joint_angles(roll_pred, joint_name="LShoulderRoll")

            # We add a sleep command to ensure the robot moves to the respective poistion before continuing 
            rospy.sleep(2.0)
            rate.sleep()

    # each Subscriber is handled in its own thread
    #rospy.spin()

if __name__=='__main__':
    parser = ArgumentParser("Program for Tutorial 3")
    parser.add_argument("--training_mode", default=False, action="store_true")
    parser.add_argument("--reset_mode", default=False, action="store_true")
   

    args = parser.parse_args()
    if args.reset_mode and args.training_mode:
        raise Exception("You can not pass both training mode and reset mode in arguments!")
    
    # instantiate class and start loop function
    print(args.training_mode)
    central_instance = Central(train_mode=args.training_mode, reset_mode=args.reset_mode)
    central_instance.central_execute()

