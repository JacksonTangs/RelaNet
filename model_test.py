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
#from main_test import dic_rank,args


#dic_rank = args.dic_rank
#

#ans_prob_threshold = args.self.ans_prob_threshold

#self.answer_size = dic_rank * 4 + 2
byte_long = 256
question_size = 20 * 20 + 4 * 256 +  2
batch_size = 64
offset_num_idx = byte_long*4
n_gram_sort = 4
img_length = 20
one_ngram_idx = 0
two_ngram_idx = byte_long
three_ngram_idx = byte_long*2
four_ngram_idx = byte_long*3
offset_num_idx = byte_long*4

#answer_one_ngram_idx = 0
#answer_two_ngram_idx = dic_rank
#answer_three_ngram_idx = dic_rank*2
#answer_four_ngram_idx = dic_rank*3
#answer_none_idx = dic_rank*4 + 1
q_type = 2# asked 1 or 2 or 3 or 4 gram
q_type_idx =n_gram_sort * byte_long + img_length * img_length

n_gram_choose = 3

#这里定义一个随着不断batch结束后不会发生改变的数字，类似全局变量
offset_pointer_record = 0

#q_type_idx = n_gram_sort * byte_long +packet_max_length
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
        self.fc3 = nn.Linear(256, args.dic_rank*4 +2)

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
    def find_ans_in_dic(self,ans):
        dirs = './data'
        ans_list = []

        part1_filename = self.protocol+'-3 byte n-gram dictionary-small-'+str(self.dic_rank)+'.pickle'
        part2_filename = self.protocol+'-3 byte n-gram dictionary-small-'+str(self.dic_rank)+'.pickle'
        part3_filename = self.protocol+'-3 byte n-gram dictionary-small-'+str(self.dic_rank)+'.pickle'
        part4_filename = self.protocol+'-3 byte n-gram dictionary-small-'+str(self.dic_rank)+'.pickle'


        #full_filename = str(i)+' byte n-gram dictionary.pickle'
        small_filename_1 = os.path.join(dirs,part1_filename)
        small_filename_2 = os.path.join(dirs,part2_filename)
        small_filename_3 = os.path.join(dirs,part3_filename)
        small_filename_4 = os.path.join(dirs,part4_filename)

        with open(small_filename_1, 'rb') as f1:
            temp_freq_small_1 = pickle.load(f1)
        with open(small_filename_2, 'rb') as f2:
            temp_freq_small_2 = pickle.load(f2)
        with open(small_filename_3, 'rb') as f3:
            temp_freq_small_3 = pickle.load(f3)
        with open(small_filename_4, 'rb') as f4:
            temp_freq_small_4 = pickle.load(f4)
        re_temp_freq_small_1 = {v:k for k,v in temp_freq_small_1.items()}
        re_temp_freq_small_2 = {v:k for k,v in temp_freq_small_2.items()}
        re_temp_freq_small_3 = {v:k for k,v in temp_freq_small_3.items()}

        re_temp_freq_small_4 = {v:k for k,v in temp_freq_small_4.items()}

        for i in range(len(ans)):
            if self.answer_one_ngram_idx <= ans[i][0] < self.answer_two_ngram_idx:
                ans_decode = re_temp_freq_small_1[ans[i][0]-self.answer_one_ngram_idx]

            elif self.answer_two_ngram_idx <= ans[i][0] < self.answer_three_ngram_idx:
                ans_decode = re_temp_freq_small_2[ans[i][0]-self.answer_two_ngram_idx]

            elif self.answer_three_ngram_idx <= ans[i][0] < self.answer_four_ngram_idx:
                #print("ans[i][0]-self.answer_three_ngram_idx",ans[i][0]-self.answer_three_ngram_idx)
                #print("re_temp_freq_small_3",len(re_temp_freq_small_3))
                ans_decode = re_temp_freq_small_3[ans[i][0]-self.answer_three_ngram_idx]



            elif self.answer_four_ngram_idx <= ans[i][0] < self.answer_none_idx:
                ans_decode = re_temp_freq_small_4[ans[i][0]-self.answer_four_ngram_idx]
            elif ans[i][0] == self.answer_none_idx :
                ans_decode = 'None'
            else :
                print("answer decode wrong  !!!!!!!!!")
            ans_list.append((ans_decode,ans[i][1]))


        f1.close()
        f2.close()
        f3.close()
        f4.close()
        return ans_list

    def parse_qst_ans(self,img,qst,ans):
        #进来的qst的size是[64,1426],进来的ans的size是[64,201],前一个参数是问题和答案个数，后面是对应编码
        #2021-02-10 将骨干模型的前后关系进行归类,归到16个类中，具体的归类方法见PPT 02-09
        question_ans_list = []
        extract_model = []
        left_extract_model_with_ans = []
        right_extract_model_with_ans = []
        left_ans_flag = 0
        right_ans_flag = 0
        time_to_write_img_flag = 0
        offset_pointer = 0
        last_temp_img = []
        #temp_img = img
        facebook_list = []#显示出有可能产生如facebook字符串的关键三元组
        global offset_pointer_record
        global answer_flag_record_left
        global answer_flag_record_right
        global record_ans_rank_first_save_left
        global record_ans_rank_first_save_right
        offset_pointer = offset_pointer_record
        #print("offset_pointer_record =",offset_pointer_record)
        #1000-fuzz-5-test-2
        #self.ans_prob_threshold
        with  open(f'./softmax-result/'+self.protocol+'-'+str(self.dic_rank)+'-fuzz-'+str(self.fuzz_range)+'-ans_prob_threshold-'+str(self.ans_prob_threshold)+'-inference_input.pickle', 'ab') as inference_f:
            with open(f'./softmax-result/'+self.protocol+'-'+str(self.dic_rank)+'-fuzz-'+str(self.fuzz_range)+'-ans_prob_threshold-'+str(self.ans_prob_threshold)+'-soft-test.csv', 'a') as log_file:
                csv_writer = csv.writer(log_file, delimiter=',')
                for i in range(batch_size):
                    #先parse问题部分
                    temp_img = img[i]
                    if i > 0:
                        last_temp_img = img[i-1]
                        #last_temp_off
                    temp_img_one_dim = np.array(temp_img)#拆成一维数组,供方便遍历
                    temp_img_one_dim = temp_img_one_dim.flatten()

                    #print(qst[i][offset_num_idx : offset_num_idx + img_length*img_length].tolist())
                    temp_offset_begin   = qst[i][offset_num_idx : offset_num_idx + img_length*img_length].tolist().index(1)
                    temp_offset_end     = img_length * img_length - qst[i][offset_num_idx : offset_num_idx + img_length*img_length].tolist()[::-1].index(1) - 1


                    #print('temp_offset_begin',temp_offset_begin)
                    #print('temp_offset_end',temp_offset_end)
                    #temp_offset_accuracy =
                    if temp_offset_begin == self.OFFSET_BEGIN and temp_offset_end == self.OFFSET_END:#遇到一个新的数据包
                        offset_pointer_record = 0
                        time_to_write_img_flag =1
                        offset_pointer = -1
                        answer_flag_record_left = 0
                        answer_flag_record_right = 0
                        record_ans_rank_first_save_left = 0
                        record_ans_rank_first_save_right = 0
                        csv_writer.writerows(question_ans_list)
                        csv_writer.writerows(left_extract_model_with_ans)
                        csv_writer.writerows(facebook_list)
                        csv_writer.writerows(right_extract_model_with_ans)
                        joblib.dump((last_temp_img,left_extract_model_with_ans,right_extract_model_with_ans), inference_f)
                        question_ans_list = []
                        extract_model = []
                        left_extract_model_with_ans = []
                        right_extract_model_with_ans = []
                        facebook_list = []


                    if time_to_write_img_flag ==1 :
                        for j in range(len(temp_img)):
                            csv_writer.writerow(list(map(lambda x: hex(int(x)).split('x')[1].zfill(2), temp_img[j])))
                        time_to_write_img_flag = 0


                    temp_one_byte = qst[i][one_ngram_idx  : one_ngram_idx  + byte_long].tolist().index(1)
                    temp_two_byte = qst[i][two_ngram_idx  : two_ngram_idx  + byte_long].tolist().index(1)
                    temp_three_byte = qst[i][three_ngram_idx  : three_ngram_idx  + byte_long].tolist().index(1)
                    #temp_q_type   = qst[i][q_type_idx     : q_type_idx     + q_type].tolist().index(1)
                    #找到那个位置
                    if i%2 == 0:
                        for k in range(offset_pointer+1,len(temp_img_one_dim)):
                            if temp_img_one_dim[k:k+n_gram_choose].tolist() == [temp_one_byte,temp_two_byte,temp_three_byte]:
                                offset_pointer = k
                                break
                    #print('extract_model: ',extract_model)
                    #print('left_extract_model_with_ans: ',left_extract_model_with_ans)
                    if i%2 ==0 :
                        if extract_model != []:

                            left_last_temp_gram = left_extract_model_with_ans[-1][0]
                            left_last_temp_offset= left_extract_model_with_ans[-1][1]
                            #left_extract_model_with_ans.append(((temp_one_byte,temp_two_byte,temp_three_byte),offset_pointer,left_relation,left_ans_flag))

                            left_now_temp_gram = (temp_one_byte,temp_two_byte,temp_three_byte)
                            left_now_temp_offset = offset_pointer

                            if (left_last_temp_gram[1] == left_now_temp_gram[0]) \
                            and(left_last_temp_gram[2] == left_now_temp_gram[1]) and \
                            (left_now_temp_offset-left_last_temp_offset==1)   :#第一种情况
                                if (left_extract_model_with_ans[-1][3] == 0) and (left_ans_flag == 0):#前确定后也确定
                                    left_relation = 1
                                elif (left_extract_model_with_ans[-1][3] == 0) and (left_ans_flag == 1):#前确定后不确定
                                    left_relation = 2

                                elif(left_extract_model_with_ans[-1][3] == 1) and (left_ans_flag == 0):#前不确定后确定
                                    left_relation = 3

                                elif(left_extract_model_with_ans[-1][3] == 1) and (left_ans_flag == 1):#前不确定后也不确定
                                    left_relation = 4
                                else:
                                    print("Some thing wrong in relation!!!!")



                            elif (left_last_temp_gram[2] == left_now_temp_gram[0]) \
                            and (left_now_temp_offset-left_last_temp_offset==2):#第二种情况
                                if (left_extract_model_with_ans[-1][3] == 0) and (left_ans_flag == 0):#前确定后也确定
                                    left_relation = 5
                                elif (left_extract_model_with_ans[-1][3] == 0) and (left_ans_flag == 1):#前确定后不确定
                                    left_relation = 6

                                elif(left_extract_model_with_ans[-1][3] == 1) and (left_ans_flag == 0):#前不确定后确定
                                    left_relation = 7

                                elif(left_extract_model_with_ans[-1][3] == 1) and (left_ans_flag == 1):#前不确定后也不确定
                                    left_relation = 8
                                else:
                                    print("Some thing wrong in relation!!!!")

                            elif left_now_temp_offset-left_last_temp_offset==3:#第三种情况
                                if (left_extract_model_with_ans[-1][3] == 0) and (left_ans_flag == 0):#前确定后也确定
                                    left_relation = 9
                                elif (left_extract_model_with_ans[-1][3] == 0) and (left_ans_flag == 1):#前确定后不确定
                                    left_relation = 10

                                elif(left_extract_model_with_ans[-1][3] == 1) and (left_ans_flag == 0):#前不确定后确定
                                    left_relation = 11

                                elif(left_extract_model_with_ans[-1][3] == 1) and (left_ans_flag == 1):#前不确定后也不确定
                                    left_relation = 12
                                else:
                                    print("Some thing wrong in relation!!!!")

                            else:#第四种情况
                                if (left_extract_model_with_ans[-1][3] == 0) and (left_ans_flag == 0):#前确定后也确定
                                    left_relation = 13
                                elif (left_extract_model_with_ans[-1][3] == 0) and (left_ans_flag == 1):#前确定后不确定
                                    left_relation = 14

                                elif(left_extract_model_with_ans[-1][3] == 1) and (left_ans_flag == 0):#前不确定后确定
                                    left_relation = 15

                                elif(left_extract_model_with_ans[-1][3] == 1) and (left_ans_flag == 1):#前不确定后也不确定
                                    left_relation = 16
                                else:
                                    print("Some thing wrong in relation!!!!")
                        else:
                            left_relation = 0#代表这时是起始位置，还没有到16个类之中


                    elif i%2 !=0:
                        if extract_model != []:

                            right_last_temp_gram = right_extract_model_with_ans[-1][0]
                            right_last_temp_offset= right_extract_model_with_ans[-1][1]

                            right_now_temp_gram = (temp_one_byte,temp_two_byte,temp_three_byte)
                            right_now_temp_offset = offset_pointer

                            if (right_last_temp_gram[1] == right_now_temp_gram[0]) \
                            and(right_last_temp_gram[2] == right_now_temp_gram[1]) and \
                            (right_now_temp_offset-right_last_temp_offset==1)   :#第一种情况
                                if (right_extract_model_with_ans[-1][3] == 0) and (right_ans_flag == 0):#前确定后也确定
                                    right_relation = 1
                                elif (right_extract_model_with_ans[-1][3] == 0) and (right_ans_flag == 1):#前确定后不确定
                                    right_relation = 2

                                elif(right_extract_model_with_ans[-1][3] == 1) and (right_ans_flag == 0):#前不确定后确定
                                    right_relation = 3

                                elif(right_extract_model_with_ans[-1][3] == 1) and (right_ans_flag == 1):#前不确定后也不确定
                                    right_relation = 4
                                else:
                                    print("Some thing wrong in relation!!!!")



                            elif (right_last_temp_gram[2] == right_now_temp_gram[0]) \
                            and (right_now_temp_offset- right_last_temp_offset==2):#第二种情况
                                if (right_extract_model_with_ans[-1][3] == 0) and (right_ans_flag == 0):#前确定后也确定
                                    right_relation = 5
                                elif (right_extract_model_with_ans[-1][3] == 0) and (right_ans_flag == 1):#前确定后不确定
                                    right_relation = 6

                                elif(right_extract_model_with_ans[-1][3] == 1) and (right_ans_flag == 0):#前不确定后确定
                                    right_relation = 7

                                elif(right_extract_model_with_ans[-1][3] == 1) and (right_ans_flag == 1):#前不确定后也不确定
                                    right_relation = 8
                                else:
                                    print("Some thing wrong in relation!!!!")

                            elif right_now_temp_offset-right_last_temp_offset==3:#第三种情况
                                if (right_extract_model_with_ans[-1][3] == 0) and (right_ans_flag == 0):#前确定后也确定
                                    right_relation = 9
                                elif (right_extract_model_with_ans[-1][3] == 0) and (right_ans_flag == 1):#前确定后不确定
                                    right_relation = 10

                                elif(right_extract_model_with_ans[-1][3] == 1) and (right_ans_flag == 0):#前不确定后确定
                                    right_relation = 11

                                elif(right_extract_model_with_ans[-1][3] == 1) and (right_ans_flag == 1):#前不确定后也不确定
                                    right_relation = 12
                                else:
                                    print("Some thing wrong in relation!!!!")

                            else:#第四种情况
                                if (right_extract_model_with_ans[-1][3] == 0) and (right_ans_flag == 0):#前确定后也确定
                                    right_relation = 13
                                elif (right_extract_model_with_ans[-1][3] == 0) and (right_ans_flag == 1):#前确定后不确定
                                    right_relation = 14

                                elif(right_extract_model_with_ans[-1][3] == 1) and (right_ans_flag == 0):#前不确定后确定
                                    right_relation = 15

                                elif(right_extract_model_with_ans[-1][3] == 1) and (right_ans_flag == 1):#前不确定后也不确定
                                    right_relation = 16
                                else:
                                    print("Some thing wrong in relation!!!!")
                        else:
                            right_relation = 0#代表这时是起始位置，还没有到16个类之中

                    '''
                    if extract_model != []:
                        last_temp_gram = temp_img_one_dim[last_offset_pointer:last_offset_pointer+n_gram_choose]#分别是上一个
                        now_temp_gram = temp_img_one_dim[offset_pointer:offset_pointer+n_gram_choose]
                    else:
                    last_offset_pointer = offset_pointer
                    '''



                    #temp_question = "What is value this {0} n-gram {2} field in offset {1}? "\
                    if qst[i][q_type_idx+1] == 1 and qst[i][q_type_idx] == 0:
                        temp_question = "What is on the right side of this {0} n-gram {2} {3} {4} field may be in left offset {1} to {5}?"\
                            .format(3, temp_offset_begin,hex(temp_one_byte),hex(temp_two_byte),hex(temp_three_byte),temp_offset_end)  # 设置指定位置
                    elif qst[i][q_type_idx+1] == 0 and qst[i][q_type_idx] == 1:
                        temp_question = "What is on the right side of this {0} n-gram {2} {3} {4} field may be in right offset {1} to {5}?"\
                            .format(3, temp_offset_begin,hex(temp_one_byte),hex(temp_two_byte),hex(temp_three_byte),temp_offset_end)  # 设置指定位置
                    else :
                        print("something wrong !!!!!!!!")


                    #print(temp_question)
                    #再parse问题部分
                    temp_ans = ans[i].tolist()
                    ans_rank_ten = sorted(enumerate(temp_ans),key = lambda x:x[1],reverse = True)[0:10]
                    ans_rank_ten = self.find_ans_in_dic(ans_rank_ten)
                    question_ans_list.append((temp_question,ans_rank_ten))
                    #left right 分开



                    if qst[i][q_type_idx+1] == 1 and qst[i][q_type_idx] == 0:

                        if extract_model == [] and offset_pointer_record == 0:
                            left_ans_flag = 0

                            #left_extract_model_with_ans.append(((hex(temp_one_byte),hex(temp_two_byte),hex(temp_three_byte)),left_ans_flag))
                            left_extract_model_with_ans.append(((temp_one_byte,temp_two_byte,temp_three_byte),offset_pointer,left_relation,left_ans_flag))

                        elif extract_model == [] and offset_pointer_record != 0:
                            left_ans_flag = answer_flag_record_left
                            ans_rank_first_save_left = record_ans_rank_first_save_left
                            if ans_rank_first_save_left == (str(temp_one_byte),str(temp_two_byte),str(temp_three_byte)):
                                wrong_flag = 0#ans_flag dont have to change
                            else :
                                wrong_flag = 1
                                if left_ans_flag == 0:
                                    left_relation = left_relation +1
                                left_ans_flag = 1
                            left_extract_model_with_ans.append(((temp_one_byte,temp_two_byte,temp_three_byte),offset_pointer,left_relation,left_ans_flag))
                            #print("ans_rank_first_save_left: ",ans_rank_first_save_left)
                            #print("(temp_one_byte,temp_two_byte,temp_three_byte): ",(str(temp_one_byte),str(temp_two_byte),str(temp_three_byte)))


                            #left_extract_model_with_ans.append(((hex(temp_one_byte),hex(temp_two_byte),hex(temp_three_byte)),left_ans_flag))

                        else:
                            if ans_rank_first_save_left == (str(temp_one_byte),str(temp_two_byte),str(temp_three_byte)):
                                #print("in it")
                                wrong_flag = 0#
                            else :
                                wrong_flag = 1
                                if left_ans_flag == 0:
                                    left_relation = left_relation +1
                                left_ans_flag = 1#ans flag直接变为错误
                            left_extract_model_with_ans.append(((temp_one_byte,temp_two_byte,temp_three_byte),offset_pointer,left_relation,left_ans_flag))
                            #print("ans_rank_first_save_left: ",ans_rank_first_save_left)
                            #print("(temp_one_byte,temp_two_byte,temp_three_byte): ",(str(temp_one_byte),str(temp_two_byte),str(temp_three_byte)))
                        if ans_rank_ten[0][1] < self.ans_prob_threshold:

                            left_ans_flag = 1#证明这个答案不太确定
                            ans_rank_first_save_left = ans_rank_ten[0][0]
                        else :

                            #这块想在加一个对答案正确性的判断
                            #即使很肯定答案，也有可能是很肯定的错误答案
                            left_ans_flag = 0#prove answer make sure
                            ans_rank_first_save_left = ans_rank_ten[0][0]


                    elif qst[i][q_type_idx+1] == 0 and qst[i][q_type_idx] == 1:#这个地方offset还没有定好是从左计算还是从右，暂时是左

                        if extract_model == [] and offset_pointer_record == 0 :
                            right_ans_flag = 0
                            #right_extract_model_with_ans.append(((hex(temp_one_byte),hex(temp_two_byte),hex(temp_three_byte)),right_ans_flag))
                            right_extract_model_with_ans.append(((temp_one_byte,temp_two_byte,temp_three_byte),offset_pointer,right_relation,right_ans_flag))
                        elif extract_model == [] and offset_pointer_record != 0:
                            right_ans_flag = answer_flag_record_right
                            ans_rank_first_save_right = record_ans_rank_first_save_right
                            if ans_rank_first_save_right == (str(temp_one_byte),str(temp_two_byte),str(temp_three_byte)):
                                wrong_flag = 0#ans_flag dont have to change
                            else :
                                wrong_flag = 1
                                if right_ans_flag == 0:
                                    right_relation = right_relation +1
                                right_ans_flag = 1
                            right_extract_model_with_ans.append(((temp_one_byte,temp_two_byte,temp_three_byte),offset_pointer,right_relation,right_ans_flag))
                        else:
                            #right_extract_model_with_ans.append(((hex(temp_one_byte),hex(temp_two_byte),hex(temp_three_byte)),right_ans_flag))
                            if ans_rank_first_save_right == (str(temp_one_byte),str(temp_two_byte),str(temp_three_byte)):
                                wrong_flag = 0#
                            else :
                                wrong_flag = 1
                                if right_ans_flag == 0:
                                    right_relation = right_relation +1
                                right_ans_flag = 1#ans flag直接变为错误


                            right_extract_model_with_ans.append(((temp_one_byte,temp_two_byte,temp_three_byte),offset_pointer,right_relation,right_ans_flag))
                        if ans_rank_ten[0][1] < self.ans_prob_threshold:
                            right_ans_flag = 1#证明这个答案不太确定
                            ans_rank_first_save_right = ans_rank_ten[0][0]
                        else :
                            right_ans_flag = 0#prove answer make sure
                            ans_rank_first_save_right = ans_rank_ten[0][0]

                        extract_model.append((hex(temp_one_byte),hex(temp_two_byte),hex(temp_three_byte)))
                    else :
                        print("something wrong !!!!!!!!")
                    #print('code end extract_model: ',extract_model)
                    #print('code end left_extract_model_with_ans: ',left_extract_model_with_ans)
                    #print("left_extract_model_with_ans",left_extract_model_with_ans)
                    #print("right_extract_model_with_ans",left_extract_model_with_ans)
                    #print("extract_model",extract_model)
                    if ans_rank_ten[0][1] > 0.95 and len(extract_model) > 1:#写facebooklist的部分
                        #print("yes")
                        #print(ans_rank_ten[0][0])
                        #print(question_ans_list[-2][1][0][0])
                        if (ans_rank_ten[0][0][0] == question_ans_list[-2][1][0][0][1]) and (ans_rank_ten[0][0][1] == question_ans_list[-2][1][0][0][2]) :

                            facebook_list.append((question_ans_list[-2][1][0][0],ans_rank_ten[0][0]))


                question_ans_list  = np.array(question_ans_list)
                question_ans_list = question_ans_list.reshape(len(question_ans_list),2)
                #print(ans_rank_ten)

                #把骨干print出来


                csv_writer.writerows(question_ans_list)
                csv_writer.writerows(left_extract_model_with_ans)
                #csv_writer.writerows(facebook_list)
                csv_writer.writerows(right_extract_model_with_ans)
                joblib.dump((temp_img,left_extract_model_with_ans,right_extract_model_with_ans), inference_f)
            log_file.close()


        inference_f.close()

        offset_pointer_record = offset_pointer
        answer_flag_record_left = left_ans_flag
        answer_flag_record_right = right_ans_flag




        record_ans_rank_first_save_left =    ans_rank_first_save_left
        record_ans_rank_first_save_right =    ans_rank_first_save_right
        return offset_pointer,time_to_write_img_flag,question_ans_list,extract_model,facebook_list,left_extract_model_with_ans,right_extract_model_with_ans


    def init_parameters(self,args):
        self.dic_rank = args.dic_rank
        self.fuzz_range  = args.fuzz_range
        self.protocol = args.protocol_type
        self.answer_size = self.dic_rank * 4 + 2
        self.answer_one_ngram_idx = 0
        self.answer_two_ngram_idx = self.dic_rank
        self.answer_three_ngram_idx = self.dic_rank*2
        self.answer_four_ngram_idx = self.dic_rank*3
        self.answer_none_idx = self.dic_rank*4 + 1
        self.ans_prob_threshold = args.ans_prob_threshold
        self.OFFSET_BEGIN = 0
        self.OFFSET_END = self.fuzz_range


    def test_(self, input_img, input_qst, label,args):
        #,right_extract_model_with_ans = self.parse_qst_ans(temp_img_batch,input_qst,np.exp(output.cuda().data.cpu()))


        self.init_parameters(args)
        output = self(input_img, input_qst)

        #print('output',np.exp(output.cuda().data.cpu()))
        #print('Output size is ',output.size())
        #print('input_img size is ',np.swapaxes(input_img.cuda().data.cpu(), 1, 3)[0].view(img_length,img_length).size())
        #print('input_img size is ',np.swapaxes(input_img.cuda().data.cpu(), 1, 3).size())
        #print('input_qst size is ',input_qst.size())
        #with open(f'./softmax-result/soft-test.csv', 'a') as log_file:
        #    csv_writer = csv.writer(log_file, delimiter=',')
            #csv_writer.writerow(['epoch',  'test_acc_norel'])

            #print(f"Training {args.model} {f'({args.relation_type})' if args.model == 'RN' else ''} model...")

        temp_img = (np.swapaxes(input_img.cuda().data.cpu(), 1, 3)[0]*255).view(img_length,img_length).tolist()
        temp_img_batch = (np.swapaxes(input_img.cuda().data.cpu(), 1, 3)*255).view(batch_size,img_length,img_length).tolist()

        #temp_img = (np.swapaxes(input_img.data.cpu(), 1, 3)[0]*255).view(img_length,img_length).tolist()
        #temp_img_batch = (np.swapaxes(input_img.data.cpu(), 1, 3)*255).view(batch_size,img_length,img_length).tolist()


        #print('temp_img size is ',np.array(temp_img).size())
        #print('temp_img size is ',len(temp_img))
        offset_pointer,time_to_write_img_flag,write_question,\
        extract_model,facebook_list,left_extract_model_with_ans\
        ,right_extract_model_with_ans = self.parse_qst_ans(temp_img_batch,input_qst,np.exp(output.cuda().data.cpu()))
        #,right_extract_model_with_ans = self.parse_qst_ans(temp_img_batch,input_qst,np.exp(output.data.cpu()))


        #print(temp_img)
        '''
        if time_to_write_img_flag ==1 :
            for i in range(len(temp_img)):
                csv_writer.writerow(list(map(lambda x: hex(int(x)).split('x')[1].zfill(2), temp_img[i])))
                #csv_writer.writerow(temp_img[i])
        '''
        #csv_writer.writerows(write_question)
        #csv_writer.writerows(left_extract_model_with_ans)
        #csv_writer.writerows(facebook_list)
        #csv_writer.writerows(right_extract_model_with_ans)
        #csv_writer.writerow(np.exp(output.cuda().data.cpu()))
        #log_file.close()




        #将答案也存一个pickle里



        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))


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
        #input可能是64 x 1 x 20 x20


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

