import argparse
#import pickle
#from tqdm import tqdm
import sys

sys.path.extend(['../'])

#import numpy as np
import os

#import xml.dom.minidom
from xml.etree import ElementTree as ET

def gdata(data_path, out_path):

    clipedEdges = 0 
    handLeftConfidence = 0 
    handLeftState = 0 
    handRightConfidence = 0 
    handRightState = 0 
    isResticted = 0 
    leanX = 0 
    leanY = 0 
    trackingState = 0

    depthX = 0
    depthY = 0
    colorX = 0 
    colorY = 0 
    orientationW = 0 
    orientationX = 0 
    orientationY = 0 
    orientationZ = 0 
    trackingState = 0
    
    action_names = []
    subjects = []
    
    action_map = {"hand_shake":"58", "high_five":"61", "fist_bump":"62", "pass_object":"56", "explain_route":"63", "rock_paper_scissors":"64", "thumbs_up":"65", "hug":"55"}
    
    for filename in os.listdir(data_path):
        fni = os.path.join(data_path, filename)
        fi=ET.parse(fni)
        root = fi.getroot()
        video = root.find('video')
        video_id = video.find('id').text
        video_name = video.find('image_data').find('path').text
        video_name = video_name[0:len(video_name)-1]
        #print(video_id)
        numFrames = int(video.find('frames_amount').text)
        frames = video.find('frames')
        action_idx = -1
        zeroBodyFrames = 0
        validActionFrames = 0
        for frame in frames:
            numBody = frame.find('skeletons_amount').text          
            if numBody == '0':
                zeroBodyFrames += 1
                continue
            skeletons = frame.find('skeletons')
            skeleton = skeletons[0]
            action_name = skeleton.find('action_name').text
            if action_name != "stand" and action_name != "approach" and action_name != "leave":
                validActionFrames += 1
                if action_name in action_names:
                    action_idx = action_names.index(action_name)
                else:
                    action_idx = len(action_names)
                    action_names.append(action_name)

        #fn = 'S000C00' + str(int(video_id) % 3 + 1) + 'P001R001A' + str(1+action_idx).zfill(3)+ '_' + video_name + '.skeleton'
        #fn = 'S000C00' + str(int(video_id) % 3 + 1) + 'P001R001A' + str(61+action_idx).zfill(3)+ '_' + video_name + '.skeleton'
        fn = 'S000C00' + str(int(video_id) % 3 + 1) + 'P001R001A' + action_map[action_names[action_idx]].zfill(3)+ '_' + video_name + '.skeleton'
        fno = os.path.join(out_path, fn)    
        fo = open(fno, "w")
        numFrames -= zeroBodyFrames
        fo.write(str(validActionFrames)+'\n')
        #fo.write(str(numFrames)+'\n')
        
        for frame in frames:
            numBody = frame.find('skeletons_amount').text          
            if numBody == '0':
                continue
            skeletons = frame.find('skeletons')
            action_name = skeletons[0].find('action_name').text
            if action_name not in action_names:
                continue
            fo.write(str(numBody)+'\n')
            for skeleton in skeletons:
                bodyID = skeleton.find('id').text
                if bodyID not in subjects:
                    subjects.append(bodyID)
                fo.write('{} {} {} {} {} {} {} {} {} {}\n'.format(bodyID, clipedEdges, handLeftConfidence, handLeftState, handRightConfidence, handRightState, isResticted, leanX, leanY, trackingState))
                numJoint = skeleton.find('joints_amount').text
                fo.write(numJoint+'\n')
                joints = skeleton.find('joints')
                for joint in joints:
                    xyz = joint.find('point3Dd').text.split()
                    fo.write('{} {} {} {} {} {} {} {} {} {} {} {}\n'.format(xyz[0], xyz[1], xyz[2], depthX, depthY, colorX, colorY, orientationW, orientationX, orientationY, orientationZ, trackingState))
        fo.close    
    print(len(action_names)) #8
    print(action_names) #['hand_shake', 'high_five', 'fist_bump', 'pass_object', 'explain_route', 'rock_paper_scissors', 'thumbs_up', 'hug']
    print(subjects) #['0', '5', '2', '1', '3', '4']
   
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ShakeFive Data Cvter.')
    parser.add_argument('--data_path', default='../data/ShakeFive_r/')
    #parser.add_argument('--out_folder', default='../data/nturgbd_raw/nturgb+d_skeletons')
    parser.add_argument('--out_folder', default='../data/ShakeFive_raw/')

    arg = parser.parse_args()

    out_path = arg.out_folder
    if not os.path.exists(out_path):
       os.makedirs(out_path)
    gdata(arg.data_path, out_path)
