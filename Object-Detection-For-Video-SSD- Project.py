# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 20:38:19 2019

@author: Tamer
"""
# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining the function that will do the detection

# Define a detect function that will take 3 arguments
# 1- frame: the orginal images from video
# 2- net: ssd neural network
# 3- transform: a transformation to be applied on images, and that will 
# return the frame with the detector rectangle

def detect(frame, net, transform):
    # Get the height and width of the frame
    # frame.shape[:2] = frame.shape[0, 1] 0 corresponding to height and 1 corresponding to width
    height, width = frame.shape[:2]
    # Apply the transformation to frame to go from original image "frame" 
    # to torch variable that will be accepted in the ssd neural netwrok
    # First transformation to get right dimentions and right colors in numpy array 
    # [0] to get only the first element of this function which is the transformed frame
    frame_t = transform(frame)[0] 
    # Transform numpy array returned "frame_t" to torch tensor which are more 
    # advanced matrix of single type
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # indexs of colors from RBG to GRB
    
    # third transormation to add fake dimension corresponding to the batch and the 
    # reson for doing this is that neural network can not accept single inputs 
    # like a single input vector or a single input image it only accept them in 
    # to some batches so now we will create a structure with first dimension 
    # corresponding to the batch and the other dimention corresponding to the input
    # this is always done with torch by using unsqueeze function
    # x.unsqueeze(0) # 0 is the index of first dimension corresponding to the batch that we add to our structure input image
    
    # Last trsformation is to convert this batch of torch tensor output into torch variable
    # Note: torch Variable is a highly advance variable that contains both tensor and gradient
    # this torch Variable will become an element of dynamic graph which will compute very effecintly 
    # the gradient of any compostion functions during backward propagation
    x = Variable(x.unsqueeze(0)) # x now is the torch Variable by using Variable class
    
    # Feed ssd neural networkwith the input images which are now torch variables
    y = net(x)
    
    # Create new tensor to take output y to become the output we intersted in
    detections = y.data # Now we got the values of the output
    
    # Create new tensor object that will have the dimensions the points of rectangle
    # This will normalize the scale values of the position of the detected object between 0 and 1
    scale = torch.Tensor([width, height, width, height])
    
    # What does detections tensor contain?
    # detections contain 4 elements [batch, number of classes, number of occurance of the class, (score, x0, y0, x1, y1)]
    
    for i in range(detections.size(1)): # detections.size(1) number of classes
        j = 0 # number of occurances of detected class
        # Keep occurances for score greater than or equal to 0.6
        while detections[0, i, j, 0] >= 0.6: # This last 0 is the index of score meansthe score of occurance j for classes i
            pt = (detections[0, i, j, 1:] * scale).numpy() # Keeping the occurance by keeping the points of the rectangle
            # 1: means x0, y0, x1, y1
            # * scale for normalization of coordinates between 0 and 1
            # numpy() function is used to convert torch tensor to numpy array
            # to allow OpenCV to draw rectangle using numpy array
            
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            # labelmap shortcut of VOC_CLASSES which is a dictionary that 
            # maps names of classes with numbers which allow us to get the 
            # label which we interested in by i-1 which is the index of the calss
            # next argument is label position next argument is the font type
            # next argument is the font size, next text color then text thickness
            # next argument to choose continues line not dotted
            
            j += 1
    return frame

# Create the ssd neural network
# test only because it's pretrained model so no need for training the model
net = build_ssd('test')

# Load the weights of ssd neural network
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

# Creating the transformation
# First argument net.size is the target size of the images to feed to the neural network
# Second argument is tuple with 3 numbers to get the right scale to make sure that color values at the right svale
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# Doing object detection on video
reader = imageio.get_reader('Original-Video.mp4')

# Get the number of frame per second fps
fps = reader.get_meta_data()['fps']

# Create output video qith that same fps
writer = imageio.get_writer('Detected-Video.mp4', fps = fps)

for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    