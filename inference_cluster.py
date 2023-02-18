#这段作为一个中转，将inference以batch为划分大小的sample cluster一下
import numpy as np
import csv
import pickle
import os
#from sklearn.externals import joblib
import joblib
import argparse

def judge_relation(last_tuple,next_tuple):
    #do judging the relation between two batches
    #arg1:last_tuple
    #arg2:next_tuple
    left_last_temp_gram = last_tuple[0]
    left_last_temp_offset= last_tuple[1]
    #left_now_temp_gram = (temp_one_byte,temp_two_byte,temp_three_byte)
    #left_now_temp_offset = offset_pointer
    left_now_temp_gram = next_tuple[0]
    left_now_temp_offset = next_tuple[1]
    if (left_last_temp_gram[1] == left_now_temp_gram[0]) \
    and(left_last_temp_gram[2] == left_now_temp_gram[1]) and \
    (left_now_temp_offset-left_last_temp_offset==1)   :#第一种情况
        if (last_tuple[3] == 0) and (next_tuple[3] == 0):#前确定后也确定
            left_relation = 1
        elif (last_tuple[3] == 0) and (next_tuple[3] == 1):#前确定后不确定
            left_relation = 2

        elif(last_tuple[3] == 1) and (next_tuple[3] == 0):#前不确定后确定
            left_relation = 3

        elif(last_tuple[3] == 1) and (next_tuple[3] == 1):#前不确定后也不确定
            left_relation = 4
        else:
            print("Some thing wrong in relation!!!!")

    elif (left_last_temp_gram[2] == left_now_temp_gram[0]) \
    and (left_now_temp_offset-left_last_temp_offset==2):#第二种情况
        if (last_tuple[3] == 0) and (next_tuple[3] == 0):#前确定后也确定
            left_relation = 5
        elif (last_tuple[3] == 0) and (next_tuple[3] == 1):#前确定后不确定
            left_relation = 6

        elif(last_tuple[3] == 1) and (next_tuple[3] == 0):#前不确定后确定
            left_relation = 7

        elif(last_tuple[3] == 1) and (next_tuple[3] == 1):#前不确定后也不确定
            left_relation = 8
        else:
            print("Some thing wrong in relation!!!!")

    elif left_now_temp_offset-left_last_temp_offset==3:#第三种情况
        if (last_tuple[3] == 0) and (next_tuple[3] == 0):#前确定后也确定
            left_relation = 9
        elif (last_tuple[3] == 0) and (next_tuple[3] == 1):#前确定后不确定
            left_relation = 10

        elif(last_tuple[3] == 1) and (next_tuple[3] == 0):#前不确定后确定
            left_relation = 11

        elif(last_tuple[3] == 1) and (next_tuple[3] == 1):#前不确定后也不确定
            left_relation = 12
        else:
            print("Some thing wrong in relation!!!!")

    else:#第四种情况
        if (last_tuple[3] == 0) and (next_tuple[3] == 0):#前确定后也确定
            left_relation = 13
        elif (last_tuple[3] == 0) and (next_tuple[3] == 1):#前确定后不确定
            left_relation = 14

        elif(last_tuple[3] == 1) and (next_tuple[3] == 0):#前不确定后确定
            left_relation = 15

        elif(last_tuple[3] == 1) and (next_tuple[3] == 1):#前不确定后也不确定
            left_relation = 16
        else:
            print("Some thing wrong in relation!!!!")

    return ((left_now_temp_gram,left_now_temp_offset,left_relation,next_tuple[3]))



parser = argparse.ArgumentParser()
parser.add_argument('--fuzz', type = int,  help='GAT with sparse version or not.')
parser.add_argument('--dict', type=int, default=72, help='Random seed.')
parser.add_argument('--threshold', type=float, default=10000, help='Number of epochs to train.')
parser.add_argument('--protocol', type=str, default='dns', help='Number of epochs to train.')
args = parser.parse_args()

#args.protocol+'-'+str(args.dict)+'-fuzz-'+str(args.fuzz)+'-ans_prob_threshold-'+str(args.threshold)+'-inference_input.pickle'
input_file = '/content/drive/MyDrive/Colab Notebooks/softmax-result/'+args.protocol+'-'+str(args.dict)+'-fuzz-'+str(args.fuzz)+'-ans_prob_threshold-'+str(args.threshold)+'-inference_input.pickle'
output_file = '/content/drive/MyDrive/Colab Notebooks/softmax-result/'+args.protocol+'-'+str(args.dict)+'-fuzz-'+str(args.fuzz)+'-ans_prob_threshold-'+str(args.threshold)+'-inference_input_cluster.pickle'
'''
if os.path.exists(input_file):
    os.remove(input_file)
'''
if os.path.exists(output_file):
    os.remove(output_file)

inference_f = open(output_file, 'ab')
pkf_wait = []
left_extract_model_with_ans_wait = []
right_extract_model_with_ans_wait =[]
write_flag = 0
pkfile2=open(input_file,'rb')
pkf,left_extract_model_with_ans,right_extract_model_with_ans=pickle.load(pkfile2)
test_packet_count = 0
try:
    while True:

        #print(count)
        pkf,left_extract_model_with_ans,right_extract_model_with_ans=pickle.load(pkfile2)
        if left_extract_model_with_ans == []:
            pkf,left_extract_model_with_ans,right_extract_model_with_ans=pickle.load(pkfile2)

        if pkf_wait == pkf:
            temp_last_left_tuple = left_extract_model_with_ans_wait[-1]
            temp_last_right_tuple = right_extract_model_with_ans_wait[-1]
            temp_next_left_tuple = left_extract_model_with_ans[0]
            temp_next_right_tuple = right_extract_model_with_ans[0]
            temp_next_left_tuple = judge_relation(temp_last_left_tuple,temp_next_left_tuple)
            temp_next_right_tuple = judge_relation(temp_last_right_tuple,temp_next_right_tuple)
            left_extract_model_with_ans[0] = temp_next_left_tuple
            right_extract_model_with_ans[0] = temp_next_right_tuple
            left_extract_model_with_ans_wait.extend(left_extract_model_with_ans)
            right_extract_model_with_ans_wait.extend(right_extract_model_with_ans)
            write_flag = 1
        else:
            if write_flag == 1:
                test_packet_count = test_packet_count +1
                joblib.dump((pkf_wait,left_extract_model_with_ans_wait,right_extract_model_with_ans_wait), inference_f)
                write_flag = 0
                print(left_extract_model_with_ans_wait)
                print(pkf_wait)
            pkf_wait = pkf
            left_extract_model_with_ans_wait = left_extract_model_with_ans
            right_extract_model_with_ans_wait = right_extract_model_with_ans

            #print(left_extract_model_with_ans)
            #pkf = np.array(pkf)
            #row_new = pkf.flatten()
            #print(right_extract_model_with_ans)
            #print(pkf)
            if left_extract_model_with_ans == []:
                print("what")
                break
except:
    print("Load Finish !!!")
    print("All packet number is ",test_packet_count," !!!!")
    pkfile2.close()
    inference_f.close()



