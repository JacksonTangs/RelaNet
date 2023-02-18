

"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
from __future__ import print_function
import argparse
import os
#import cPickle as pickle
import pickle
import random
import numpy as np
import csv

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from model_test import RN, CNN_MLP
#from model import
#from sklearn.externals import joblib
import joblib
import time

file_train_num = 1
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN',
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--relation-type', type=str, default='binary',
                    help='what kind of relations to learn. options: binary, ternary (default: binary)')
parser.add_argument('--dic_rank', type=int, default=1000, metavar='N',
                    help='dictionary size choose by model (default: 1000)')
parser.add_argument('--ans_prob_threshold', type=float, default=0.97, metavar='LR',
                    help='Answer problem threshold (default: 0.97)')
parser.add_argument('--fuzz_range', type=int, default=5, metavar='N',
                    help='fuzz range of the question (default: 5)')
parser.add_argument('--protocol_type', type=str,default='dns',
                    help='protocol type')




args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
dic_rank = args.dic_rank #子字典的建立是选取大字典的前*位
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

summary_writer = SummaryWriter()

if args.model=='CNN_MLP':
  model = CNN_MLP(args)
else:
  model = RN(args)

model_dirs = './model'
bs = args.batch_size
#dic_rank = 1000 #子字典的建立是选取大字典的前*位
answer_size = dic_rank * 4 + 2
img_size = 20
question_size = img_size * img_size + 4 * 256 +  2
input_img = torch.FloatTensor(bs, 1, img_size, img_size)
input_qst = torch.FloatTensor(bs, question_size)
label = torch.LongTensor(bs)
#label = torch.FloatTensor(bs)

if args.cuda:
    model.cuda()
    input_img = input_img.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()

input_img = Variable(input_img)
input_qst = Variable(input_qst)
label = Variable(label)

def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)


def test(epoch):
    model.eval()
    '''
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return

    '''
    #norel = cvt_data_axis(norel)
    accuracy_norels = []
    loss_unary = []
    average_time = 0
    with open(f'./time_save.csv', 'a') as f_time:
        csv_writer = csv.writer(f_time, delimiter=',')
        csv_writer.writerow(['packet_num','average_time'])
        for file_count in range(0,file_train_num):
            #_,norel = load_data('test-'+str(file_count+1)+'.pickle',1)
            _,norel = load_data(args.protocol_type+'-'+str(dic_rank)+'-fuzz-'+str(args.fuzz_range)+'-test-'+str(file_count+1)+'.pickle',1)

            norel = cvt_data_axis(norel)

            for batch_idx in range(len(norel[0]) // bs):
                tensor_data(norel, batch_idx)
                beg = time.time()
                acc_un, l_un = model.test_(input_img, input_qst, label,args)
                end_time = time.time()-beg
                average_time = (end_time/64+average_time)/2
                accuracy_norels.append(acc_un.item())
                loss_unary.append(l_un.item())

            csv_writer.writerow([(file_count+1)*100,average_time])

        accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
        print('\n  Test set : Unary accuracy: {:.0f}%\n'.format(
             accuracy_norel))

        summary_writer.add_scalars('Accuracy/test', {
            'unary': accuracy_norel
        }, epoch)

        loss_unary = sum(loss_unary) / len(loss_unary)

        summary_writer.add_scalars('Loss/test', {

            'unary': loss_unary
        }, epoch)
        print("average_time is: ",average_time)
        
        f_time.close()




    return accuracy_norel


def load_data(filename_part,test_or_train):
    print('loading data...')
    dirs = './data'

    filename = os.path.join(dirs,filename_part)
    if test_or_train == 0:
        with open(filename, 'rb') as f:
          train_datasets, test_datasets = joblib.load(f)
    elif test_or_train == 1 :
        with open(filename, 'rb') as f:
          _,test_datasets = joblib.load(f)
    else:
        print("something wrong!!")

    norel_train = []
    norel_test = []
    print('processing data...')
    if test_or_train == 0:
        for img,  norelations in train_datasets:
            img = np.swapaxes(img, 0, 2)
            for qst,ans in zip(norelations[0], norelations[1]):
                norel_train.append((img,qst,ans))
        print('processing train finish')

    elif test_or_train == 1:
        for img,  norelations in test_datasets:
            img = np.swapaxes(img, 0, 2)
            for qst,ans in zip(norelations[0], norelations[1]):
                norel_test.append((img,qst,ans))
        print("processing test finish")
    else:
        print("something wrong!!")
    f.close()

    return ( norel_train, norel_test)


try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

#这块应该是要加一个循环load数据
#for i in range
#norel_train, norel_test = load_data()



if args.resume:
    filename = os.path.join(model_dirs, args.resume)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

with open(f'./{args.model}_{args.seed}_only-test_log.csv', 'w') as log_file:
    csv_writer = csv.writer(log_file, delimiter=',')
    csv_writer.writerow(['epoch',  'test_acc_norel'])

    print(f"Training {args.model} {f'({args.relation_type})' if args.model == 'RN' else ''} model...")


    #或者是在这里加上众多数据集

    #norel_train,norel_test = load_data()
    test_acc_unary = test(1)
    csv_writer.writerow([1,test_acc_unary])
        #model.save_model(epoch)
