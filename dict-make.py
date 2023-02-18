#############################################################
#目标是另存为一个小字典，这个小字典是计数字典中前50个键值对##
#      12-25开会:50个键值对太少，取1000个      ##
#   01-07 :新方案，先取50个              ##
#   01-12方案，2gram 1000个              ##
#   01-30方案 3gram 择出高频有意义词汇        ##
#   02-07   择出一些 3 c o这样的3 gram       ##
#   05-07   处理ntp流量，择出高频年份         ##
#############################################################
import os
import pickle
import operator
import argparse
dic_freq_count = 0

#字典备选值 600 800 1000 1200
#consider_freq_num = 1000 # 出现在频率字典中的MsgType备选值，要被考虑问问题的前这么多个
threshold_num = 600#一个门限值，如果高于这个值则让它择出字典最低的里头,这样的话基本不会在字典里
new_rank_freq = dict()
ascii_num_range = range(48,58)#ascii数字对应十进制范围
ascii_upper_range = range(65,91)#ascii大写字母对应十进制范围A-Z
ascii_lower_range = range(97,123)#ascii小写字母对应十进制范围a-z
#print('loading data...')
def making_small_dictionary(protocol,consider_freq_num):
    dirs = '/content/drive/MyDrive/Colab Notebooks/data'
    print("Here is top ",consider_freq_num,"dictionary !!")
    #windows_size = [1,2,3,4]
    windows_size = [3]#只做3gram的
    for i in windows_size:
        new_rank_freq = dict()
        dic_freq_count = 0
        part_filename = protocol+'-'+str(i)+' byte n-gram dictionary.pickle'
        filename = os.path.join(dirs,part_filename)
        #filename = os.path.join(dirs,str(i)+' byte n-gram dictionary.pickle')
        with open(filename, 'rb') as f:
            temp_freq,temp_offset_left,temp_offset_right = pickle.load(f)

        f.close()
        #d2 = sorted(temp_freq.items(), key=lambda d:d[1],reverse = True)
        d2 = dict(sorted(temp_freq.items(), key=operator.itemgetter(1),reverse = True))

        print(d2)
        #以下部分添加于01-30,目的是把高频变长择出来,如com
        #02-07修改：去除如 3 c o 的 3gram长度字段
        for key,value in d2.items():
            temp_key_count  = 0
            num_key_count = 0
            char_key_count = 0
            #print('key: ',key)
            

            for temp_key in key:
                #if (int(temp_key) in ascii_num_range) or (int(temp_key) in ascii_upper_range) or (int(temp_key) in ascii_lower_range):#如果n-gram组里东西都在范围内，挑选出来
                if int(temp_key) in (list(ascii_num_range)+list(ascii_upper_range)+list(ascii_lower_range)):#如果n-gram组里东西都在范围内，挑选出来
                #把对应的东西择出来
                    #并且这个排名靠前
                    temp_key_count = temp_key_count + 1

                    #print('temp_key_count : ',temp_key_count)
                    if protocol == 'dns':
                        if dic_freq_count <= threshold_num and temp_key_count >= len(key) -1:#两个在ascii里头的就可以
                            #print("get out!!!")
                            d2[key] = 1#将其次数降为非常低的情况
                #这部分想去除 3 c o 的情况 o m 2    m 2 c
                '''
                if (int(temp_key) in ascii_num_range) or (int(temp_key) in ascii_upper_range) or (int(temp_key) in ascii_lower_range):#如果n-gram组里东西都在范围内，挑选出来
                #把对应的东西择出来
                    #并且这个排名靠前
                    temp_key_count = temp_key_count + 1

                    #print('temp_key_count : ',temp_key_count)
                    if dic_freq_count <= threshold_num and temp_key_count == len(key):
                        #print("get out!!!")
                        d2[key] = 1#将其次数降为非常低的情况
                '''

            dic_freq_count = dic_freq_count +1
        dic_freq_count = 0
        d3 = dict(sorted(d2.items(), key=operator.itemgetter(1),reverse = True))
        for key,value in d3.items():

            print(key,value)
            new_rank_freq[key] = dic_freq_count
            dic_freq_count = dic_freq_count + 1
            if dic_freq_count >= consider_freq_num:
                break

        #print(new_rank_freq)
        print("Sum of the dictionay elementary is ",dic_freq_count,"dictionary !!")
        part_filename = protocol+'-'+str(i)+' byte n-gram dictionary-small-'+str(consider_freq_num)+'.pickle'
        filename = os.path.join(dirs,part_filename)
        #print(filename)
        with  open(filename, 'wb') as f:
            pickle.dump(new_rank_freq, f)

        f.close()
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol_type', type=str, default='dns',
                    help='protocol type (default: dns)')
    parser.add_argument('--dict', type=int, default=1000,
                    help='dictionary number (default: 1000)')

    args = parser.parse_args()
    protocol = args.protocol_type

    #part_filename = protocol+'-'+str(i)+' byte n-gram dictionary.pickle'



    making_small_dictionary(protocol,args.dict)
    #making_small_dictionary(800)

    #making_small_dictionary(600)
    #making_small_dictionary(1200)
    #making_small_dictionary(1400)

