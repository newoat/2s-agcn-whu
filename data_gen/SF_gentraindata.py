import argparse
import pickle
from tqdm import tqdm
import sys

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 505###300

import numpy as np
import os


def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())#first line: number of frames
        #print(file)
        #print(f"\nnumFrame: {skeleton_sequence['numFrame']}")
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    #print(skeleton_sequence)
    #print(f"skeleton_sequence: {skeleton_sequence}")
    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):  #first 4 body and 25 joints.
    seq_info = read_skeleton_filter(file)#def read_skeleton_filter(file):
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    #fst = True;
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]#m: body,   n: frames,   j: joints.
                    #only x,y,z
                    
                    #if fst:
                    #    print([v['x'], v['y'], v['z']])
                    #    fst = False;
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])#standard deviation
    index = energy.argsort()[::-1][0:max_body_true]#max_body_true = 2
    print(f"index: {index}")#[0 3]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        
        if action_class > 60: #60=>11, 61=>12
            continue
        #action_class -= 49 #55->6, 56->7, 58 -> 9 action_map = {"hand_shake":"58", "pass_object":"56", "hug":"55"}
    
        #subject_id = int(
        #    filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        #istraining = (camera_id != 1)
        istraining = (camera_id!=4)

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
        #def read_xyz(file, max_body=4, num_joint=25):
        fp[i, :, 0:data.shape[1], :, :] = data

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ShakeFive Data Converter.')
    parser.add_argument('--data_path', default='../data/ShakeFive_raw_train/')
    ###
    parser.add_argument('--out_folder', default='../data/ShakeFiveP2_train/')###
    #parser.add_argument('--out_folder', default='../data/ntu/')###
    ###

    benchmark = ['xview']#
    part = ['train', 'val']###
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                benchmark=b,
                part=p)
