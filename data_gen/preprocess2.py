import sys

sys.path.extend(['../'])
from data_gen.rotation import *
from tqdm import tqdm


def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]): #0,1:spine. #4,8:shoulders.
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C
    #print('s.shape:   ', s.shape) #(N,2,400,25,3)
    print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(tqdm(s)):  # pad
        #print('skeleton.shape:   ', skeleton.shape)  #(2,400,25,3)
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            #print(person.shape) #(400,25,3)
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                #print(frame.shape) #(25,3)
                ###aaatest()
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break

    print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1]) #i_s: num of files. for each file, cal an axis and angle.
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    print(
        'parallel the joint1 person1 and joint1 person2 to the y axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue    
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        ax = joint_rshoulder - joint_lshoulder
        ax0 = [1, 0, 0]
        ##########################################
        if 1:
            joint_rshoulder = skeleton[1, 0, 0]
            joint_lshoulder = skeleton[0, 0, 0]
            ax = joint_rshoulder - joint_lshoulder
            ax = [ax[0],ax[1],0]
            ax0 = [0,1,0]
        ##########################################    
        axis = np.cross(ax, ax0)
        angle = angle_between(ax, ax0)
        matrix_x = rotation_matrix(axis, angle)             
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])#N,M,T,V,C backto N,C,T,V,M 
    return data


if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)
