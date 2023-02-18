import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import csv
#from sklearn.externals import joblib
import joblib
import os
import pickle
#dic_rank = 1000 #子字典的建立是选取大字典的前*位
#answer_size = dic_rank * 4 + 2
byte_long = 256
question_size = 20 * 20 + 4 * 256 +  2
batch_size = 64
offset_num_idx = 0

img_length = 20
one_ngram_idx = img_length * img_length


answer_one_ngram_idx = 0
#answer_two_ngram_idx = dic_rank
#answer_three_ngram_idx = dic_rank*2#
#answer_four_ngram_idx = dic_rank*3
#answer_none_idx = dic_rank*4
q_type = 2# asked 1 or 2 or 3 or 4 gram
q_type_idx = byte_long + img_length * img_length
class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)

        #self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        #self.batchNorm3 = nn.BatchNorm2d(24)
        #self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        #self.batchNorm4 = nn.BatchNorm2d(24)


    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        #x = self.conv3(x)
        #x = F.relu(x)
        #x = self.batchNorm3(x)
        #x = self.conv4(x)
        #x = F.relu(x)
        #x = self.batchNorm4(x)
        return x


class FCOutputModel(nn.Module):
    def __init__(self,args):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        #self.fc3 = nn.Linear(256, 10)
        self.fc3 = nn.Linear(256, args.dictionary_choose*4+2)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)

        #loss_fn = nn.MSELoss(reduce = False,size_average = False)

        #loss = loss_fn(output, label)

        #temp_label = torch.FloatTensor(label.data.cpu().numpy())
        #temp_label = temp_label.cuda()
        #loss.backward(torch.ones_like(label))
        loss.backward()

        self.optimizer.step()
        #print('Output size is ',output.size())
        #print('label size is ',label.size())
        pred = output.data.max(1)[1]
        #print('pred size is ',pred.size())
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss
    '''
    def init_parameters(self,args):
        self.dic_rank = args.dic_rank
        self.answer_size = self.dic_rank * 4 + 2
    '''
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        #print('output',np.exp(output.cuda().data.cpu()))
        #print('Output size is ',output.size())
        #print('input_img size is ',np.swapaxes(input_img.cuda().data.cpu(), 1, 3)[0].view(img_length,img_length).size())
        #print('input_qst size is ',input_qst.size())

        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def save_model(self, epoch,args):
        torch.save(self.state_dict(), 'model/'+args.protocol_type+'-'+str(args.dictionary_choose)+'-fuzz-'+str(args.fuzz_range)+'epoch_{}_{:02d}.pth'.format(self.name, epoch))
        #args.protocol_type+'-'+str(dictionary_choose)+'-fuzz-'+str(fuzz_range)+'


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')

        self.conv = ConvInputModel()

        self.relation_type = args.relation_type

        if self.relation_type == 'ternary':
            ##(number of filters per object+coordinate of object)*3+question vector
            self.g_fc1 = nn.Linear((24+2)*3+question_size, 256)
        else:
            ##(number of filters per object+coordinate of object)*2+question vector
            self.g_fc1 = nn.Linear((24+2)*2+question_size, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]

        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.fcout = FCOutputModel(args)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)

        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2)


        if self.relation_type == 'ternary':
            # add question everywhere
            qst = torch.unsqueeze(qst, 1) # (64x1x18)
            qst = qst.repeat(1, 25, 1) # (64x25x18)
            qst = torch.unsqueeze(qst, 1)  # (64x1x25x18)
            qst = torch.unsqueeze(qst, 1)  # (64x1x1x25x18)

            # cast all triples against each other
            x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26)
            x_i = torch.unsqueeze(x_i, 3)  # (64x1x25x1x26)
            x_i = x_i.repeat(1, 25, 1, 25, 1)  # (64x25x25x25x26)

            x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26)
            x_j = torch.unsqueeze(x_j, 2)  # (64x25x1x1x26)
            x_j = x_j.repeat(1, 1, 25, 25, 1)  # (64x25x25x25x26)

            x_k = torch.unsqueeze(x_flat, 1)  # (64x1x25x26)
            x_k = torch.unsqueeze(x_k, 1)  # (64x1x1x25x26)
            x_k = torch.cat([x_k, qst], 4)  # (64x1x1x25x26+18)
            x_k = x_k.repeat(1, 25, 25, 1, 1)  # (64x25x25x25x26+18)

            # concatenate all together
            x_full = torch.cat([x_i, x_j, x_k], 4)  # (64x25x25x25x3*26+18)

            # reshape for passing through network
            x_ = x_full.view(mb * (d * d) * (d * d) * (d * d), 96)  # (64*25*25*25x3*26+18) = (1.000.000, 96)
        else:
            # add question everywhere
            qst = torch.unsqueeze(qst, 1)
            qst = qst.repeat(1, 25, 1)
            qst = torch.unsqueeze(qst, 2)

            # cast all pairs against each other
            x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26+18)
            x_i = x_i.repeat(1, 25, 1, 1)  # (64x25x25x26+18)
            x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26+18)
            x_j = torch.cat([x_j, qst], 3)
            x_j = x_j.repeat(1, 1, 25, 1)  # (64x25x25x26+18)

            # concatenate all together
            x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+18)

            # reshape for passing through network
            x_ = x_full.view(mb * (d * d) * (d * d), question_size+2*26)  # (64*25*25*(2*26+18)) = (40.000, 70)

        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)

        # reshape again and sum
        if self.relation_type == 'ternary':
            x_g = x_.view(mb, (d * d) * (d * d) * (d * d), 256)
        else:
            x_g = x_.view(mb, (d * d) * (d * d), 256)

        x_g = x_g.sum(1).squeeze()

        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)

        return self.fcout(x_f)


class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv  = ConvInputModel()
        self.fc1   = nn.Linear(5*5*24 + 18, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        #print([ a for a in self.parameters() ] )

    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)

        x_ = torch.cat((x, qst), 1)  # Concat question

        x_ = self.fc1(x_)
        x_ = F.relu(x_)

        return self.fcout(x_)

