import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module): #temporal GCN
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module): #spatial GCN
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3): #3 subsets: self, centripetal, centrifugal
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels # i_c: Cin, o_c: Cout, inter_c= Ce
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))#PA:B #change tensor to trainable parameter
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False) #astype: Copy of the array, cast to a specified type. #torch.from_numpy: Creates a Tensor from a numpy.ndarray. #Variable: wraps a Tensor.
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential( #residual box is only needed when Cin is not the same as Cout
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size() # = N * M, C, T, V
        A = self.A.cuda(x.get_device()) #A
        A = A + self.PA #PA: Bk #Ak+Bk

        y = None
        for i in range(self.num_subset):#kv=3
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T) #theta_k
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V) #phi_k
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1)) #Ck #N V V #torch.matmul: Matrix product of two tensors #size(-1): the last dimension
            A1 = A1 + A[i] # = Ak + Bk + Ck. #A1: Ck. #N*N
            A2 = x.view(N, C * T, V) #CinT*N
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V)) #omega_k
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x) #+residual box
        return self.relu(y)

class TCN_GCN_unit(nn.Module): #AGC Block
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True): #In the constructor, you declare all the layers you want to use.
        super(TCN_GCN_unit, self).__init__() #father class
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0 #lambda function is a small anonymous function.

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x): #In the forward function, you define how your model is going to be run, from input to output
        x = self.tcn1(self.gcn1(x)) + self.residual(x) #Fig 3.
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3): #in_channels = x,y,z
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # adjacency_matrix, shape = (3,25,25)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False) #B1
        self.l2 = TCN_GCN_unit(64, 64, A) #B2
        self.l3 = TCN_GCN_unit(64, 64, A) #B2.5
        self.l4 = TCN_GCN_unit(64, 64, A) #B3
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2) #B4
        self.l6 = TCN_GCN_unit(128, 128, A) #B5
        self.l7 = TCN_GCN_unit(128, 128, A) #B6
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2) #B7
        self.l9 = TCN_GCN_unit(256, 256, A) #B8
        self.l10 = TCN_GCN_unit(256, 256, A) #B9

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size() #N=samples. C=channel,xyz,3. T,temporal length=frames. V=joints,25. M=skeletons,max=2.

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T) #permute(0, 4, 3, 1, 2): N,M,V,C,T. #.view: Returns a new tensor with the same data as the self tensor but of a different shape.
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V) #permute(0, 1, 3, 4, 2): N,M,C,T,V

        x = self.l1(x) #outsize = NM,64,T,V
        x = self.l2(x) #outsize = NM,64,T,V
        x = self.l3(x) #outsize = NM,64,T,V
        x = self.l4(x) #outsize = NM,64,T,V
        x = self.l5(x) #outsize = NM,128,T/2,V
        x = self.l6(x) #outsize = NM,128,T,V
        x = self.l7(x) #outsize = NM,128,T,V
        x = self.l8(x) #outsize = NM,256,T/4,V
        x = self.l9(x) #outsize = NM,256,T/4,V
        x = self.l10(x) #outsize = NM,256,T/4,V

        # N*M,C,T,V
        c_new = x.size(1) #new channel, 256
        x = x.view(N, M, c_new, -1) #-1:actual value for this dimension will be inferred. #N,M,256,TV/4
        x = x.mean(3).mean(1) #mean(3)??? ave of TV? #mean(1): average of 2 person. #N,256

        return self.fc(x) #N,60
