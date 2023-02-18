#1
####################################
#先制作抽象模型，再生成问题及答案###
#给每个网络DNS流量产生问题及答案新##
#     把数据包修建补0版本         ##
#     给问题和答案编码版本        ##
#    01-12版本,问右边2n-gram    ##
#    01-21版本，问右边3n-gram   ##
#    01-29版本,双向问3 n-gram   ##
####################################
import csv
import pickle
import os
import numpy as np
#from sklearn.externals import joblib
import joblib
import argparse
dirs = '/content/drive/MyDrive/Colab Notebooks/data'
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


def abstract_model_making(fuzz_range,dictionay_choose,freq_dic_small_3,n_gram_choose,protocol):
    #不同协议此处的值不同
    #DNS 42
    #modbus 54
    if protocol == 'dns':
        open_path = '/content/drive/MyDrive/Colab Notebooks/data/01-21-top-1500.csv'
        front_Length = 0  #ethernet,IP,UDP length
    elif protocol == 'modbus':
        open_path = '/content/drive/MyDrive/Colab Notebooks/data/ics-modbus-pure.csv'
        front_Length = 0  #ethernet,IP,UDP length
    elif protocol == 'dnp3':
        open_path = '/content/drive/MyDrive/Colab Notebooks/data/dnp3-pure.csv'
        front_Length = 0  #ethernet,IP,UDP length
    elif protocol == 'dhcp':
        open_path = '/content/drive/MyDrive/Colab Notebooks/data/dhcp-pure.csv'
        front_Length = 0  #ethernet,IP,UDP length
    elif protocol == 'ntp':
        open_path = '/content/drive/MyDrive/Colab Notebooks/data/ntp-pure.csv'
        front_Length = 0  #ethernet,IP,UDP length
    else:
        print("we dont have this protocol!!")

    #front_Length = 54#ethernet,IP,UDP length
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

    dic_rank = dictionay_choose #子字典的建立是选取大字典的前*位

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

    q_1 = Queue()
    q_2 = Queue()
    q_3 = Queue()
    q_4 = Queue()
    zero_judge = Queue()
    count = 0
    file_count = 1

    fuzz_slice = np.ones(2*fuzz_range).astype(int)
    #open_path = '/content/drive/MyDrive/Colab Notebooks/data/dns_new.csv'
    #open_path = '/content/drive/MyDrive/Colab Notebooks/data/01-21-top-1500.csv'
    #open_path = '/content/drive/MyDrive/Colab Notebooks/data/modbus-top-6000.csv'

    with open(open_path,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            extract_model = []
            question_list = []
            answer_list = []
            zero_judge.__init__()
            offset_count = 0
            packet_id = packet_id + 1
            temp_dic = dict()
            #print(row[front_Length:])
            row_length = len(row[front_Length:])
            if (len(row[front_Length:])) < packet_max_length:
                row_new = ['0' for x in range(0,packet_max_length)]
                row_new[:len(row[front_Length:])] = row[front_Length:]
            else:
                row_new = row[front_Length:packet_max_length]

            question_bin_list = []
            answer_bin_list = []
            #print(row_new)
            row_img = np.zeros((packet_max_length))
            for num in row_new:
                #print(num)
                row_img[offset_count] = int(num)
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
                    if temp_tuple_3 in freq_dic_small_3:#判断扫到的这个2 n-gram在这个小字典里:
                        if temp_tuple_3 == ('0', '0','0'):#数据包尾部补充的0不提问题
                            zero_judge.enqueue((temp_tuple_3,offset_count -4,row_length -(offset_count -4)))
                        else :
                            if zero_judge.is_empty():
                                extract_model.append((temp_tuple_3,offset_count -4,row_length -(offset_count -4)))
                            else :
                                for i in range(zero_judge.size()):
                                    extract_model.append(zero_judge.dequeue())
                                extract_model.append((temp_tuple_3,offset_count -4,row_length -(offset_count -4)))

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
            for i in range(0,q_1.size()):
                q_1.dequeue()
            for i in range(0,q_2.size()):
                q_2.dequeue()
            for i in range(0,q_3.size()):
                q_3.dequeue()
            for i in range(0,q_4.size()):
                q_4.dequeue()
            #print(extract_model)
            if int(extract_model[0][0][1]) == 1 and int(extract_model[0][0][1]) == 0 :
                print("Something bad happened!!!")
            for i in range(len(extract_model)):

                temp_tuple = extract_model[i][0]
                if len(extract_model[i][0]) == 3:
                    question_bin_right = np.zeros((question_size))
                    question_bin_left = np.zeros((question_size))

                    if i-fuzz_range < 0:
                        offset_left_bound_right = offset_num_idx
                    else:
                        offset_left_bound_right = offset_num_idx + i-fuzz_range
                    if i+fuzz_range > packet_max_length:
                        offset_right_bound_right = packet_max_length + offset_num_idx
                    else :
                        #offset_right_bound_right = offset_num_idx + i+fuzz_range
                        offset_right_bound_right = offset_num_idx + i+fuzz_range+1#因为fuzz为0的原因 这块多加个1

                    if row_length <= packet_max_length:#有可能包会比截的400长,需要分别处理
                        if (len(extract_model) - i) - fuzz_range < 0:
                            offset_left_bound_left = offset_num_idx
                        else:
                            offset_left_bound_left = offset_num_idx + (len(extract_model) - i)-fuzz_range
                        if (len(extract_model) - i)+fuzz_range > packet_max_length:
                            offset_right_bound_left = packet_max_length + offset_num_idx
                        else :
                            #offset_right_bound_left = offset_num_idx + (len(extract_model) - i)+fuzz_range
                            offset_right_bound_left = offset_num_idx + (len(extract_model) - i)+fuzz_range+1#因为fuzz为0的原因 这块多加个1
                    else:
                        temp_extract = (len(extract_model) - i)%packet_max_length
                        if temp_extract-fuzz_range < 0:
                            offset_left_bound_left = offset_num_idx
                        else:
                            offset_left_bound_left = offset_num_idx + temp_extract-fuzz_range
                        if temp_extract+fuzz_range > packet_max_length:
                            offset_right_bound_left = packet_max_length + offset_num_idx
                        else :
                            #offset_right_bound_left = offset_num_idx + temp_extract+fuzz_range
                            offset_right_bound_left = offset_num_idx + temp_extract+fuzz_range+1#因为fuzz为0的原因 这块多加个1


                    #底下的right和left为朝向，比如说right是指从左到右的offset
                    question_bin_right [offset_left_bound_right:offset_right_bound_right] = 1
                    temp_question_right = "What is on the right side of this {0} n-gram {2} {3} {4}field may be with left offset {1} to {5}? "\
                                            .format(n_gram_choose, offset_left_bound_right - offset_num_idx,hex(int(temp_tuple[0]))\
                                                   ,hex(int(temp_tuple[1])),hex(int(temp_tuple[2])),offset_right_bound_right - offset_num_idx)  # 设置指定位置

                    question_bin_left [offset_left_bound_left:offset_right_bound_left] = 1
                    temp_question_left = "What is on the right side of this {0} n-gram {2} {3} {4}field may be with right offset {1} to {5}? "\
                                            .format(n_gram_choose, offset_left_bound_left - offset_num_idx,hex(int(temp_tuple[0]))\
                                                   ,hex(int(temp_tuple[1])),hex(int(temp_tuple[2])),offset_right_bound_left - offset_num_idx)  # 设置指定位置


                    # question content part
                    question_bin_right [one_ngram_idx + int(temp_tuple[0])] = 1
                    question_bin_right [two_ngram_idx + int(temp_tuple[1])] = 1
                    question_bin_right [three_ngram_idx + int(temp_tuple[2])] = 1
                    question_bin_right [q_type_idx + 1] = 1



                    # question content part
                    question_bin_left [one_ngram_idx + int(temp_tuple[0])] = 1
                    question_bin_left [two_ngram_idx + int(temp_tuple[1])] = 1
                    question_bin_left [three_ngram_idx + int(temp_tuple[2])] = 1
                    question_bin_left [q_type_idx] = 1

                    #print(temp_question_left)
                    #print(temp_question_right)

                else:
                    print("something happend wrong")


                answer_dec_right = 0

                if i == len(extract_model)-1:
                    temp_answer_right = 'None'
                    #answer_bin_right[answer_none_idx + 1] = 1
                    answer_dec_right = answer_none_idx + 1
                else :
                    temp_answer_right = extract_model[i+1][0]
                    if len(temp_answer_right) == n_gram_choose:
                        #answer_bin_left[ answer_two_ngram_idx  +freq_dic_small_2[temp_answer_left]  ] = 1
                        answer_dec_right = answer_three_ngram_idx+freq_dic_small_3[temp_answer_right]

                    else:
                        print("right Answer is wrong !!!!!!!!!!!!!")

                #print(temp_question_right,temp_answer_right)
                #print(question_bin_right,answer_dec_right)
                answer_list.append(answer_dec_right)
                question_list.append(question_bin_right)

                answer_list.append(answer_dec_right)
                question_list.append(question_bin_left)
            row_img  = row_img/255.

            dataset_packet = (np.array(row_img).reshape(packet_img_size,packet_img_size,1),(question_list,answer_list))
            print("Dealing with "+str(packet_id)+" packet !!!")
            if packet_id <= train_size:
                if file_count > 0:
                    train_datasets.append(dataset_packet)
            elif  packet_id > train_size and packet_id <= train_size+test_size:
                if file_count > 0:
                    test_datasets.append(dataset_packet)
            else :
                print('saving datasets...')
                filename_test = os.path.join(dirs,protocol+'-'+str(dictionay_choose)+'-fuzz-'+str(fuzz_range)+'-test-'+str(file_count)+'.pickle')
                filename_train = os.path.join(dirs,protocol+'-'+str(dictionay_choose)+'-fuzz-'+str(fuzz_range)+'-train-test-'+str(file_count)+'.pickle')

                if file_count > 0:
                    with  open(filename_train, 'wb') as f_train:
                        joblib.dump((train_datasets, test_datasets), f_train)#train
                        #joblib.dump(([], test_datasets), f)#test
                    f_train.close()
                    with  open(filename_test, 'wb') as f_test:
                        #joblib.dump((train_datasets, test_datasets), f)#train
                        joblib.dump(([], test_datasets), f_test)#test
                    f_test.close()
                file_count = file_count + 1
                if file_count == 6:
                    break
                print('datasets saved at {}'.format(filename_train))
                print('datasets saved at {}'.format(filename_test))
                train_datasets = []
                test_datasets = []
                packet_id = 0

def main(fuzz_range,dictionay_choose,protocol):
    #dirs = './data'
    #dirs = '/content/drive/MyDrive/Colab Notebooks/data'
    try:
        os.makedirs(dirs)
    except:
        print('directory {} already exists'.format(dirs))


    #part1_filename = '1 byte n-gram dictionary-small-'+str(dictionay_choose)+'.pickle'
    #part2_filename = '2 byte n-gram dictionary-small-'+str(dictionay_choose)+'.pickle'
    #part3_filename = '3 byte n-gram dictionary-small-'+str(dictionay_choose)+'.pickle'
    part3_filename = protocol+'-'+str(3)+' byte n-gram dictionary-small-'+str(dictionay_choose)+'.pickle'
    #protocol+'-'+str(3)+' byte n-gram dictionary-small-'+str(consider_freq_num)+'.pickle
    #part4_filename = '4 byte n-gram dictionary-small-'+str(dictionay_choose)+'.pickle'

    #part1_filename = '1 byte n-gram dictionary-small.pickle'
    #part2_filename = '2 byte n-gram dictionary-small.pickle'
    #part3_filename = '3 byte n-gram dictionary-small.pickle'
    #part4_filename = '4 byte n-gram dictionary-small.pickle'


    #full_filename = str(i)+' byte n-gram dictionary.pickle'
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


    #abstract_model_making(dictionay_choose,temp_freq_small_1,temp_freq_small_2,temp_freq_small_3,temp_freq_small_4,3)
    abstract_model_making(fuzz_range,dictionay_choose,temp_freq_small_3,3,protocol)

    #f1.close()
    #f2.close()
    f3.close()
    #f4.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol_type', type=str, default='dns',
                    help='protocol type (default: dns)')
    parser.add_argument('--dictionay_choose', type=int, default='1000',
                    help='protocol type (default: 1000)')
    parser.add_argument('--fuzz_range', type=int, default='5',
                    help='protocol type (default: 1000)')
    args = parser.parse_args()
    protocol = args.protocol_type
    #dictionay_choose = 1000
    #fuzz_range =
    main(args.fuzz_range,args.dictionay_choose,protocol)

