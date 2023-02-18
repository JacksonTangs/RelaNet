#########################################
# 02-05 完成最后一个formats inference模块#
# input: 原始payload，和模型的推理结果    #
# 上好色的原始报文                        #
#########################################
from dns_parse import parse_dns_test
from modbus_parse import parse_modbus_test
from dns_parse import deal_domain_name
from ntp_parse import parse_ntp_test
from dnp3_parse import parse_dnp3_test
from dhcp_parse import parse_dhcp_test
from numpy import *
import numpy as np
import csv
import pickle
#import panda as pd
import os
import argparse
#from sklearn.externals import joblib
import joblib

#定义两种类型字段，分别为固定字段和不固定字段
#fix_field
#var_field
#给不在字典中的报文field上色
#input从pickle来
class Queue(object):
    """队列"""
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def enqueue(self, item):
        """进队列"""
        self.items.insert(0,item)

    def dequeue(self):
        """出队列"""
        return self.items.pop()

    def size(self):
        """返回大小"""
        return len(self.items)
    def content(self):
        """返回值"""
        return self.items
    def search(self,rank):
        """查询特定位置"""
        return self.items[-(rank)]


#03-02:准备写一些标准评价系统
'''
def Precision_caculate(A,):
    #计算准确(TruePositive)/(TruePositive+FalsePositive)
def Recall_caculate():
    #计算召回率(TruePositive)/(TruePositive+FalseNegative)
def F-measure():
    #计算F-score2*(Precision*Recall)/(Precision+Recall)


def cal_correctness_perfection(y_true,y_pred):
'''
def cal_precision_recall(y_true,y_pred):
    #1代表正，0代表负
    #将固定字段看成正,不固定字段看成负
    #true positive
    TP = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,1)))
    #print("TP: ",TP)

    #false positive
    FP = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,1)))
    #print("FP: ",FP)

    #true negative
    TN = np.sum(np.logical_and(np.equal(y_true,0),np.equal(y_pred,0)))
    #print("TN: ",TN)

    #false negative
    FN = np.sum(np.logical_and(np.equal(y_true,1),np.equal(y_pred,0)))
    #print("FN: ",FN)
    accuracy = float((TP+TN)/(TP+FP+TN+FN))
    print("accuracy: ",accuracy)
    precision = float(TP/(TP+FP))
    print("precision: ",precision)
    recall  = float(TP/(TP+FN))
    print("recall: ",recall)
    F_measure = float( 2 * precision * recall /(precision+recall))
    print("F_measure: ",F_measure)
    return accuracy,precision,recall,F_measure

def not_in_dictionary(freq_dic_small_3,n_gram_choose,protocol,dic_rank,metric_threshold,fuzz_range,threshold):
    #front_Length = 42#ethernet,IP,UDP length
    packet_id = 0
    offset_count = 0
    Offset_left_count = dict()
    Offset_right_count = dict()
    temp_list = []
    byte_long = 256
    n_gram_sort  = 4
    packet_img_size = 20
    packet_max_length = packet_img_size * packet_img_size#

    write_list = []
    one_ngram_idx = 0
    two_ngram_idx = byte_long
    three_ngram_idx = byte_long*2
    four_ngram_idx = byte_long*3
    offset_num_idx = byte_long*4

    #dic_rank = 1000 #子字典的建立是选取大字典的前*位

    answer_one_ngram_idx = 0
    answer_two_ngram_idx = dic_rank
    answer_three_ngram_idx = dic_rank*2
    answer_four_ngram_idx = dic_rank*3
    answer_none_idx = dic_rank*4
    q_type = 2# asked right or left
    q_type_idx = n_gram_sort * byte_long +packet_max_length
    #question 的编码部分有什么

    #问题对象是几个byte的n-gram模型，有1 byte and 2 byte and 3 by   256 * 4 + offset()+2
    question_size = n_gram_sort * byte_long + packet_max_length +q_type
    answer_size = n_gram_sort * byte_long + 2
    #上面的2表示左到头了和右到头了
    packet_all_num = 800000#z总共处理的dns流量个数
    train_size = 500
    test_size = 100
    train_datasets = []
    test_datasets = []
    question_list = []
    answer_list = []
    write_list = []
    extract_model  =[]
    abstract_model_list = []
    abstract_model_bin_list = []
    correct_result_list = []
    precision_list = []
    recall_list = []
    F_measure_list = []
    accuracy_list = []
    regular_pattern_list = []

    q_1 = Queue()
    q_2 = Queue()
    q_3 = Queue()
    q_4 = Queue()
    zero_judge = Queue()
    count = 0
    file_count = 1
    #fuzz_range = 5
    #fuzz_slice = np.ones(2*fuzz_range).astype(int)
    left_extract_model_offset = 0
    right_extract_model_offset = 0
    print_list = str()
    begin_flag = 0
    left_extract_model_finish = 0

    #寻求一个门限值，即在算各种指标时考虑该值以前的
    #metric_threshold = 96
    # 取前多少个字节作为计算metric时的范围
    #open_path = '/content/drive/MyDrive/Colab Notebooks/data/dns_new.csv'
    #open_path = '/content/drive/MyDrive/Colab Notebooks/data/01-21-top-1500.csv'
    #file_name = '/content/drive/MyDrive/Colab Notebooks/data/dns_new.csv'
    output_file = '/content/drive/MyDrive/Colab Notebooks/softmax-result/'+protocol+'-'+str(dic_rank)+'-fuzz-'+str(fuzz_range)+'-ans_prob_threshold-'+str(threshold)+'-inference_input_cluster.pickle'
    regular_dirs = '/content/drive/MyDrive/Colab Notebooks/finger_print'
    pkfile2=open(output_file,'rb')

    try:
        while True:
            pkf,left_extract_model_with_ans,right_extract_model_with_ans=pickle.load(pkfile2)
            if left_extract_model_with_ans == []:
                pkf,left_extract_model_with_ans,right_extract_model_with_ans=pickle.load(pkfile2)
            pkf = np.array(pkf)
            row_new = pkf.flatten()
            #print(temp_pkf)
            #print(temp_pkf.shape())
            #print()
            print_list  = str()
            pcap_confirm = list()
            regular_pattern = str()
            type_flag_array = np.zeros((packet_max_length))
            left_extract_model_offset = 0
            right_extract_model_offset = 0

            for num in row_new:
                #print(num)
                num = str(int(num))
                if q_1.size() < 1 and q_2.size() < 2 and q_3.size() < 3 and q_4.size() < 4:
                    #print('douxiao')
                    q_1.enqueue(num)
                    q_2.enqueue(num)
                    q_3.enqueue(num)
                    q_4.enqueue(num)
                elif q_2.size() < 2 and q_3.size() < 3 and q_4.size() < 4:
                    q_2.enqueue(num)
                    q_3.enqueue(num)
                    q_4.enqueue(num)
                elif q_3.size() < 3 and q_4.size() < 4:
                    q_3.enqueue(num)
                    q_4.enqueue(num)
                elif q_4.size() < 4:
                    q_4.enqueue(num)

                else :
                    #q.enqueue(num)
                    #print('四个队列都填满了')
                    temp_list_4 = []
                    q_4.content().reverse()
                    for i in range(4):
                        temp_list_4.append(q_4.content()[i])
                    temp_tuple_4 = tuple(temp_list_4)
                    q_4.content().reverse()
                    #print(temp_tuple)

                    temp_list_3 = []
                    q_3.content().reverse()
                    for i in range(3):
                        temp_list_3.append(q_3.content()[i])
                    temp_tuple_3 = tuple(temp_list_3)
                    q_3.content().reverse()

                    temp_list_2 = []
                    q_2.content().reverse()
                    for i in range(2):
                        temp_list_2.append(q_2.content()[i])
                    temp_tuple_2 = tuple(temp_list_2)
                    q_2.content().reverse()

                    temp_list_1 = []
                    q_1.content().reverse()
                    for i in range(1):
                        temp_list_1.append(q_1.content()[i])
                    temp_tuple_1 = tuple(temp_list_1)
                    q_1.content().reverse()
                    #print(temp_tuple_3)
                    ##################
                    #简单化第一次上色#
                    ##################
                    '''
                    if protocol == 'dns':
                        if left_extract_model_with_ans[left_extract_model_offset][1] == 2 and begin_flag != 1:#这个地方的2是因为所有的DNS协议开始是在2开始
                            print_list  = str()
                            pcap_confirm = list()
                            begin_flag = 1
                    elif protocol == 'modbus':

                        if dic_rank > 300:
                            if left_extract_model_with_ans[left_extract_model_offset][1] == 1 and begin_flag != 1:#这个地方的是0因为所有的modbus协议开始是在2开始
                                print_list  = str()
                                pcap_confirm = list()
                                begin_flag = 1
                        else:

                        if left_extract_model_with_ans[left_extract_model_offset][1] == 2 and begin_flag != 1:#这个地方的是0因为所有的modbus协议开始是在2开始
                            print_list  = str()
                            pcap_confirm = list()
                            begin_flag = 1
                    else:
                        print("Begin wrong, dont have this protocol!!")
                    '''

                    if temp_tuple_3 in freq_dic_small_3:#判断扫到的这个3 n-gram在这个小字典里,如果是则认为是个固定字段:
                        '''
                        if left_extract_model_finish == 1 :
                            left_extract_model_finish = 0
                            break
                        '''
                        if left_extract_model_with_ans[left_extract_model_offset][2] == 0:#这种情况全黄
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[0])+' \033[0m'

                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[1])+' \033[0m'

                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[2])+' \033[0m'

                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[0]))[2:] +' '
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[1]))[2:] +' '
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[2]))[2:] +' '

                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[0])) +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[1])) +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[2])) +' '



                            type_flag_array[offset_count-4] = 2 #2是固定字段

                            type_flag_array[offset_count-3] = 2 #2是固定字段

                            type_flag_array[offset_count-2] = 2 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[0]))
                            pcap_confirm.append(int(temp_tuple_3[1]))
                            pcap_confirm.append(int(temp_tuple_3[2]))


                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 1 :#这种情况是后一个给黄，前确定，后确定
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[2])+' \033[0m'
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[2]))[2:] +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[2])) +' '
                            type_flag_array[offset_count-2] = 2 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[2]))

                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 2:#这种情况是后一个给蓝，前确定，后不确定
                            print_list = print_list + '\033[1;32;44m'+str(temp_tuple_3[2])+' \033[0m'#蓝
                            regular_pattern = regular_pattern + '(.*) '
                            type_flag_array[offset_count-2] = 1 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[2]))

                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 3 :#这种情况前不确定，后确定
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[2])+' \033[0m'
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[2]))[2:] +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[2])) +' '
                            type_flag_array[offset_count-2] = 2 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[2]))

                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 4:#这种情况前不确定，后不确定
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[2])+' \033[0m'
                            regular_pattern = regular_pattern + '(.*) '
                            type_flag_array[offset_count-2] = 1 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[2]))



                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 5:#这种情况是后两个个给黄，前确定，后确定
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[1])+' \033[0m'
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[2])+' \033[0m'
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[1]))[2:] +' '
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[2]))[2:] +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[1])) +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[2])) +' '
                            type_flag_array[offset_count-3] = 2 #2是固定字段
                            type_flag_array[offset_count-2] = 2 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[1]))
                            pcap_confirm.append(int(temp_tuple_3[2]))

                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 6:#这种情况是后一个给蓝，前确定，后不确定
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[1])+' \033[0m'
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[2])+' \033[0m'
                            regular_pattern = regular_pattern + '(.*) '
                            type_flag_array[offset_count-3] = 1 #2是固定字段
                            type_flag_array[offset_count-2] = 1 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[1]))
                            pcap_confirm.append(int(temp_tuple_3[2]))

                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 7:#这种情况前不确定，后确定
                            print_list = print_list + '\033[1;32;43m'+str(temp_tuple_3[1])+' \033[0m'
                            print_list = print_list + '\033[1;32;43m'+str(temp_tuple_3[2])+' \033[0m'
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[1]))[2:] +' '
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[2]))[2:] +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[1])) +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[2])) +' '
                            type_flag_array[offset_count-3] = 2 #2是固定字段
                            type_flag_array[offset_count-2] = 2 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[1]))
                            pcap_confirm.append(int(temp_tuple_3[2]))

                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 8:#这种情况前不确定，后不确定
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[1])+' \033[0m'
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[2])+' \033[0m'
                            regular_pattern = regular_pattern + '(.*) '
                            type_flag_array[offset_count-3] = 1 #2是固定字段
                            type_flag_array[offset_count-2] = 1 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[1]))
                            pcap_confirm.append(int(temp_tuple_3[2]))



                        #这四种是3gram挨着的情况
                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 9:#这种情况是后一个给黄，前确定，后确定
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[0])+' \033[0m'
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[1])+' \033[0m'
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[2])+' \033[0m'
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[0]))[2:] +' '
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[1]))[2:] +' '
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[2]))[2:] +' '

                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[0])) +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[1])) +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[2])) +' '
                            type_flag_array[offset_count-4] = 2 #2是固定字段

                            type_flag_array[offset_count-3] = 2 #2是固定字段

                            type_flag_array[offset_count-2] = 2 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[0]))
                            pcap_confirm.append(int(temp_tuple_3[1]))
                            pcap_confirm.append(int(temp_tuple_3[2]))

                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 10:#这种情况是后一个给蓝，前确定，后不确定
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[0])+' \033[0m'
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[1])+' \033[0m'
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[2])+' \033[0m'
                            regular_pattern = regular_pattern + '(.*) '
                            type_flag_array[offset_count-4] = 1 #2是固定字段

                            type_flag_array[offset_count-3] = 1 #2是固定字段

                            type_flag_array[offset_count-2] = 1 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[0]))
                            pcap_confirm.append(int(temp_tuple_3[1]))
                            pcap_confirm.append(int(temp_tuple_3[2]))
                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 11:#这种情况前不确定，后确定
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[0])+' \033[0m'
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[1])+' \033[0m'
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[2])+' \033[0m'
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[0]))[2:] +' '
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[1]))[2:] +' '
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[2]))[2:] +' '

                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[0])) +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[1])) +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[2])) +' '
                            type_flag_array[offset_count-4] = 2 #2是固定字段

                            type_flag_array[offset_count-3] = 2 #2是固定字段

                            type_flag_array[offset_count-2] = 2 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[0]))
                            pcap_confirm.append(int(temp_tuple_3[1]))
                            pcap_confirm.append(int(temp_tuple_3[2]))
                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 12:#这种情况前不确定，后不确定
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[0])+' \033[0m'
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[1])+' \033[0m'
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[2])+' \033[0m'
                            regular_pattern = regular_pattern + '(.*) '
                            type_flag_array[offset_count-4] = 1 #2是固定字段

                            type_flag_array[offset_count-3] = 1 #2是固定字段

                            type_flag_array[offset_count-2] = 1 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[0]))
                            pcap_confirm.append(int(temp_tuple_3[1]))
                            pcap_confirm.append(int(temp_tuple_3[2]))


                        #目前只要属于隔着的情况，都给不确定
                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 13:#这种情况是后一个给黄，前确定，后确定
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[0])+' \033[0m'
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[1])+' \033[0m'
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[2])+' \033[0m'
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[0]))[2:] +' '
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[1]))[2:] +' '
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[2]))[2:] +' '

                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[0])) +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[1])) +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[2])) +' '
                            type_flag_array[offset_count-4] = 2 #2是固定字段

                            type_flag_array[offset_count-3] = 2 #2是固定字段

                            type_flag_array[offset_count-2] = 2 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[0]))
                            pcap_confirm.append(int(temp_tuple_3[1]))
                            pcap_confirm.append(int(temp_tuple_3[2]))
                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 14:#这种情况是后一个给蓝，前确定，后不确定
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[0])+' \033[0m'
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[1])+' \033[0m'
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[2])+' \033[0m'
                            regular_pattern = regular_pattern + '(.*) '
                            type_flag_array[offset_count-4] = 1 #2是固定字段

                            type_flag_array[offset_count-3] = 1 #2是固定字段

                            type_flag_array[offset_count-2] = 1 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[0]))
                            pcap_confirm.append(int(temp_tuple_3[1]))
                            pcap_confirm.append(int(temp_tuple_3[2]))
                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 15:#这种情况前不确定，后确定
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[0])+' \033[0m'
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[1])+' \033[0m'
                            print_list = print_list+'\033[1;32;43m'+str(temp_tuple_3[2])+' \033[0m'
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[0]))[2:] +' '
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[1]))[2:] +' '
                            #regular_pattern = regular_pattern + hex(int(temp_tuple_3[2]))[2:] +' '

                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[0])) +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[1])) +' '
                            regular_pattern = regular_pattern + "{:02x}".format(int(temp_tuple_3[2])) +' '
                            type_flag_array[offset_count-4] = 2 #2是固定字段

                            type_flag_array[offset_count-3] = 2 #2是固定字段

                            type_flag_array[offset_count-2] = 2 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[0]))
                            pcap_confirm.append(int(temp_tuple_3[1]))
                            pcap_confirm.append(int(temp_tuple_3[2]))
                        elif left_extract_model_with_ans[left_extract_model_offset][2] == 16:#这种情况前不确定，后不确定
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[0])+' \033[0m'
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[1])+' \033[0m'
                            print_list = print_list+'\033[1;32;44m'+str(temp_tuple_3[2])+' \033[0m'
                            regular_pattern = regular_pattern + '(.*) '
                            type_flag_array[offset_count-4] = 1 #2是固定字段

                            type_flag_array[offset_count-3] = 1 #2是固定字段

                            type_flag_array[offset_count-2] = 1 #2是固定字段
                            pcap_confirm.append(int(temp_tuple_3[0]))
                            pcap_confirm.append(int(temp_tuple_3[1]))
                            pcap_confirm.append(int(temp_tuple_3[2]))


                        left_extract_model_offset = left_extract_model_offset+1

                        begin_flag = 0

                    else : #不在字典里,认为是个可变字段，3 gram的首个byte是可变的
                        if type_flag_array[offset_count-4] == 0:#这个字段在前面没被标记过
                            type_flag_array[offset_count-4] = 1 #1是不固定字段
                            print_list = print_list + '\033[1;32;44m'+str(temp_tuple_3[0])+' \033[0m'
                            regular_pattern = regular_pattern + '(.*) '
                            pcap_confirm.append(int(temp_tuple_3[0]))



                    if left_extract_model_offset == len(left_extract_model_with_ans):#这个代表骨干模型的所有东西都已经扫描完成
                        #left_extract_model_finish = 1#这个代表骨干模型的所有东西都已经扫描完成
                        print('len_row_new',len(row_new[offset_count-4:]))
                        print('pcap_confirm',pcap_confirm)
                        print('offset_count: ',offset_count)
                        for row_element in row_new[len(pcap_confirm):]:
                            print_list = print_list + '\033[1;32;44m'+str(int(row_element))+' \033[0m'
                            regular_pattern = regular_pattern + '(.*) '
                            pcap_confirm.append(int(row_element))
                        #print('row_new： ',list(row_new.astype(np.int16)))
                        #print('pcap_confirm ',pcap_confirm)
                        break



                    q_1.dequeue()
                    q_2.dequeue()
                    q_3.dequeue()
                    q_4.dequeue()
                    q_1.enqueue(q_2.content()[0])
                    q_2.enqueue(q_3.content()[0])
                    q_3.enqueue(q_4.content()[0])
                    q_4.enqueue(num)
                offset_count = offset_count +1
            offset_count = 0
            #print(type_flag_array)
            #print(pcap_confirm)
            #print('pcap_confirm_length:',len(pcap_confirm))
            for i in range(0,packet_max_length-len(pcap_confirm)):
                pcap_confirm.append(0)
            #print('pcap_confirm_length:',len(pcap_confirm))
            print('original pacap is: ',row_new)
            row_new = list(row_new.astype(np.int16))
            if len(pcap_confirm) >=packet_max_length:
                pcap_confirm = pcap_confirm[:packet_max_length]
            if row_new == pcap_confirm:
                print("OK")
            else:
                print("You have Problem !!!!")
                #compoare_index = np.arrange
                #print('left_extract_model_with_ans: ',left_extract_model_with_ans)
                #print('row_new： ',row_new)
                #print('pcap_confirm ',pcap_confirm)

                pcap_confirm_compare = np.array(pcap_confirm)
                row_new_compare = np.array(row_new)
                #定义数组下标
                if len(pcap_confirm_compare) == len(row_new_compare):
                    compare_index = np.arange(0,len(pcap_confirm_compare))
                else:
                    print('Verify pcap confirm length wrong !!!!')
                #找到两个数组相等元素的下标位置
                #print(index[a == b])
                #找到两个数组不相等元素的下标位置
                print(compare_index[pcap_confirm_compare != pcap_confirm_compare])

            if protocol == 'dns':
                real_extract_model,real_type_flag_array = parse_dns_test(row_new)
            elif protocol == 'modbus':
                real_extract_model,real_type_flag_array = parse_modbus_test(row_new)
            elif protocol == 'dhcp':
                real_extract_model,real_type_flag_array = parse_dhcp_test(row_new)
            elif protocol == 'ntp':
                real_extract_model,real_type_flag_array = parse_ntp_test(row_new)
            elif protocol == 'dnp3':
                real_extract_model,real_type_flag_array = parse_dnp3_test(row_new)
            else:
                print("We dont have this protocol to parse")
            real_length = len(real_type_flag_array)
            real_length = min(real_length,packet_max_length,metric_threshold)



            type_flag_array = np.array(type_flag_array[0:real_length])
            standard_print_list = str()





            for i in range(0,real_length):
                if real_type_flag_array[i] == 1:
                    standard_print_list = standard_print_list+'\033[1;32;44m'+str(row_new[i])+' \033[0m'

                elif real_type_flag_array[i] == 2:
                    standard_print_list = standard_print_list+'\033[1;32;43m'+str(row_new[i])+' \033[0m'
            #print(standard_print_list)
            #print(print_list)
            print("regular_pattern: ",regular_pattern)
            regular_pattern_list.append(regular_pattern)
            real_type_flag_array = np.array(real_type_flag_array)
            #定义数组下标
            index = np.arange(0,real_length)
            #print("here is ok")
            #找到两个数组相等元素的下标位置
            print("type_flag_array shape is ",type_flag_array.shape)
            print("real_type_flag_array shape is ",real_type_flag_array.shape)



            #num_correct_sample = len(index[type_flag_array == real_type_flag_array])
            #找到两个数组不相等元素的下标位置
            #num_incorrect_sample = len(index[type_flag_array != type_flag_array])
            #print("there is ",100*num_correct_sample/real_length,"% judge right !!!!")
            #correct_result_list.append(100*num_correct_sample/real_length)


            for i in range(0,len(type_flag_array)):
                if type_flag_array[i] == 1:
                    type_flag_array[i] = 0
                elif type_flag_array[i] == 2:
                    type_flag_array[i] = 1
                #else:
                #    print("something wrong !!!!")

            for i in range(0,len(real_type_flag_array)):
                if real_type_flag_array[i] == 1:
                    real_type_flag_array[i] = 0
                elif real_type_flag_array[i] == 2:
                    real_type_flag_array[i] = 1
                #else:
                #    print("something wrong !!!!")

            #参数前面是true 后面应该是predict
            accuracy,precision,recall,F_measure = cal_precision_recall(real_type_flag_array[0:real_length],type_flag_array[0:real_length])
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            F_measure_list.append(F_measure)
            print()
            for i in range(0,q_1.size()):
                q_1.dequeue()
            for i in range(0,q_2.size()):
                q_2.dequeue()
            for i in range(0,q_3.size()):
                q_3.dequeue()
            for i in range(0,q_4.size()):
                q_4.dequeue()
            #print(extract_model)


    except:
        print("Load Finish !!!")
        #print(correct_result_list)

        print("accuracy mean: ",mean(accuracy_list),"\n")
        #print("accuracy: ",mean(correct_result_list))
        print("precision mean: ",mean(precision_list),"\n")
        print("recall mean: ",mean(recall_list),"\n")
        print("F_measure mean: ",mean(F_measure_list),"\n")
        final_open = open('/content/drive/MyDrive/Colab Notebooks/final-result/'+protocol+'-final-result.csv','a+',newline="")
        csv_writer = csv.writer(final_open)
        csv_writer.writerow([dic_rank,fuzz_range,threshold,metric_threshold,mean(accuracy_list),mean(precision_list),mean(recall_list),mean(F_measure_list)])




        filename_finger_print = os.path.join(regular_dirs,protocol+'-finger-print.pickle')



        with  open(filename_finger_print, 'wb') as f_test:
            #joblib.dump((train_datasets, test_datasets), f)#train
            joblib.dump(regular_pattern_list, f_test)#test
        f_test.close()


        print('datasets saved at {}'.format(filename_finger_print))



        pkfile2.close()
        final_open.close()



parser = argparse.ArgumentParser()
parser.add_argument('--fuzz', type = int,default=5)
parser.add_argument('--dict', type=int, default=1000)
parser.add_argument('--threshold', type=float, default=10000)
parser.add_argument('--protocol', type=str, default='dns')
parser.add_argument('--metric_threshold', type=int, default=32)
args = parser.parse_args()

#args.protocol+'-'+str(args.dict)+'-fuzz-'+str(args.fuzz)+'-ans_prob_threshold-'+str(args.threshold)+'-inference_input.pickle'

#output_file = '/content/drive/MyDrive/Colab Notebooks/softmax-result/'+args.protocol+'-'+str(args.dict)+'-fuzz-'+str(args.fuzz)+'-ans_prob_threshold-'+str(args.threshold)+'-inference_input_cluster.pickle'




dirs = './data'
#dirs = '/content/drive/MyDrive/Colab Notebooks/data'
dirs = '/content/drive/MyDrive/Colab Notebooks/data'
try:
    os.makedirs(dirs)
except:
    print('directory {} already exists'.format(dirs))


#part1_filename = args.protocol+'-1 byte n-gram dictionary-small-'+str(args.dict)+'.pickle'
#part2_filename = args.protocol+'-2 byte n-gram dictionary-small-'+str(args.dict)+'.pickle'
part3_filename = args.protocol+'-3 byte n-gram dictionary-small-'+str(args.dict)+'.pickle'
#part4_filename = args.protocol+'-4 byte n-gram dictionary-small-'+str(args.dict)+'.pickle'


#small_filename_1 = os.path.join(dirs,part1_filename)
#small_filename_2 = os.path.join(dirs,part2_filename)
small_filename_3 = os.path.join(dirs,part3_filename)
#small_filename_4 = os.path.join(dirs,part4_filename)

'''
with open(small_filename_1, 'rb') as f1:
    temp_freq_small_1 = pickle.load(f1)
with open(small_filename_2, 'rb') as f2:
    temp_freq_small_2 = pickle.load(f2)
with open(small_filename_4, 'rb') as f4:
    temp_freq_small_4 = pickle.load(f4)
'''
with open(small_filename_3, 'rb') as f3:
    temp_freq_small_3 = pickle.load(f3)

#print(temp_freq_small_3)
not_in_dictionary(temp_freq_small_3,3,args.protocol,args.dict,args.metric_threshold,args.fuzz,args.threshold)

#f1.close()
#f2.close()
f3.close()
#f4.close()