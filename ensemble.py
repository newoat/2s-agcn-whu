import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='ntu/xview', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

######################
#dataset = 'ShakeFiveP2_tl/xview'
dataset = 'ntu_p2/xview'
######################
label = open('./data/' + dataset + '/val_label.pkl', 'rb')
label = np.array(pickle.load(label))

###########################
dataset = arg.datasets
nm1 = '/p2agcn_test_joint_p2'
nm2 = '/p2agcn_test_bone'
###########################
r1 = open('./work_dir/' + dataset + nm1 + '/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('./work_dir/' + dataset + nm2 + '/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
##################################
print(label.shape) #(2,3458)
print(type(r1)) #list
print(len(r1)) #number of val samples
print(len(r1[0])) #2
print(r1[0]) 
print(r1[0][0]) #skeleton file name
print(r1[0][1]) #result/score/weights for 11 classes #[15.896442    2.2354932  -0.24630809  1.3714168  -2.1260164  -0.44466305  0.665004   -0.8494295  -3.1023204  -8.128081   -5.2690344 ]
right_num = total_num = right_num_5 = 0
#############################################
import pandas as pd
actual = []
predicted = []
#############################################
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    if i==0:
        print()
        print(l) #actual class
        print(r11) #all classes
    r = r11 + r22 * arg.alpha
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r) #predicted class
    right_num += int(r == int(l))
    total_num += 1
    actual.append(int(l))
    predicted.append(r)
acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)
y_actu = pd.Series(actual, name='Actual')
y_pred = pd.Series(predicted, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)