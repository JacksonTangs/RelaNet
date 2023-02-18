#01-21 这个应该是做字典
import csv
import pickle
import os
import argparse
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


def dictionary_making(window_size,protocol):
    '''
    if protocol == 'dns':
        front_Length = 42  #ethernet,IP,UDP length
    elif protocol == 'modbus':
        front_Length = 66  #ethernet,IP,UDP length
    else:
        print("We dont have this protocol")
    '''
    line_count = 0
    MsgType_count = dict()
    Offset_left_count = dict()
    Offset_right_count = dict()
    temp_list = []
    #open_file = '/content/drive/MyDrive/Colab Notebooks/data/dns_new.csv'
    #open_file = '/content/drive/MyDrive/Colab Notebooks/data/01-21-top-1500.csv'
    if protocol == 'dns':
        open_file = '/content/drive/MyDrive/Colab Notebooks/data/01-21-top-1500.csv'
        front_Length = 0  #ethernet,IP,UDP length
    elif protocol == 'modbus':
        open_file = '/content/drive/MyDrive/Colab Notebooks/data/ics-modbus-pure.csv'
        front_Length = 0  #ethernet,IP,UDP length
    elif protocol == 'dnp3':
        open_file = '/content/drive/MyDrive/Colab Notebooks/data/dnp3-pure.csv'
        front_Length = 0  #ethernet,IP,UDP length
    elif protocol == 'dhcp':
        open_file = '/content/drive/MyDrive/Colab Notebooks/data/dhcp-pure.csv'
        front_Length = 0  #ethernet,IP,UDP length
    elif protocol == 'ntp':
        open_file = '/content/drive/MyDrive/Colab Notebooks/data/ntp-pure.csv'
        front_Length = 0  #ethernet,IP,UDP length
    else:
        print("we dont have this protocol!!")
    q = Queue()
    count = 0
    with open(open_file,'r') as f:
        reader = csv.reader(f)
        print(type(reader))
        for row in reader:
            print(row)
            line_count  =line_count+ 1
            if line_count % 1000 == 0:
              print("dealing with "+ str(line_count)+" lines")
            count = 0
            for num in row[front_Length:]:
                if num != '\n':
                    if q.size() < (window_size-1):
                        q.enqueue(num)
                    else :

                        q.enqueue(num)
                        temp_list = []
                        q.content().reverse()
                        #reg_q = q.content()
                        #reg_q = reg_q.reverse()
                        #print(q.content())
                        #print(type(num))
                        for i in range(window_size):
                            temp_list.append(q.content()[i])

                        #temp_tuple = (q.content()[0],q.content()[1])
                        temp_tuple = tuple(temp_list)
                        #print(temp_tuple)

                        if temp_tuple not in MsgType_count:
                            MsgType_count[temp_tuple] = 1
                        else :
                            MsgType_count[temp_tuple] = MsgType_count[temp_tuple] + 1

                        ################################################
                        if temp_tuple not in Offset_left_count:# caculate left offset
                            Offset_left_count[temp_tuple] = dict()
                            Offset_left_count[temp_tuple][count]=1
                        else :
                            #for offcount in Offset_count[num]:
                            if count not in Offset_left_count[temp_tuple].keys():
                                Offset_left_count[temp_tuple][count]=1
                            else :
                                Offset_left_count[temp_tuple][count]= Offset_left_count[temp_tuple][count]+1
                        ################################################
                        if temp_tuple not in Offset_right_count:# caculate right offset
                            Offset_right_count[temp_tuple] = dict()
                            Offset_right_count[temp_tuple][len(row[front_Length:]) - count]=1
                        else :
                            #for offcount in Offset_count[num]:
                            if (len(row[front_Length:]) - count) not in Offset_right_count[temp_tuple].keys():
                                Offset_right_count[temp_tuple][len(row[front_Length:]) - count]=1
                            else :
                                Offset_right_count[temp_tuple][len(row[front_Length:]) - count]= Offset_right_count[temp_tuple][len(row[front_Length:]) - count]+1
                        #print("dealing with "+ str(count)+" lines")
                        count = count+1

                        q.content().reverse()
                        q.dequeue()

                else:
                    for i in range(0,window_size-1):
                        #print(q.content())
                        q.dequeue()

    print("dealing " + str(window_size)+ " graph finish !!!! ")
    #print(MsgType_count)

    d2 = sorted(MsgType_count.items(), key=lambda d:d[1],reverse = True) #[('ok', 1), ('no', 2)]
    #print(d2)
    #print()
    #print()
    #print()
    #print(MsgType_count['129'])
    #print(Offset_left_count)
    #print(MsgType_count)
    return MsgType_count,Offset_left_count,Offset_right_count
    #print(Offset_left_count['129'])
    #print(Offset_right_count['129'])
    #print(MsgType_count['2211'])


if __name__ == '__main__':

    dirs = '/content/drive/MyDrive/Colab Notebooks/data'
    try:
        os.makedirs(dirs)
    except:
        print('directory {} already exists'.format(dirs))


    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol_type', type=str, default='dns',
                    help='protocol type (default: dns)')
    args = parser.parse_args()
    protocol = args.protocol_type

    windows_size = [1,2,3 , 4]


    for i in windows_size:
        part_filename = protocol+'-'+str(i)+' byte n-gram dictionary.pickle'
        filename = os.path.join(dirs,part_filename)
        temp_freq,temp_offset_left,temp_offset_right = dictionary_making(i,protocol)
        with  open(filename, 'wb') as f:
            pickle.dump((temp_freq,temp_offset_left,temp_offset_right), f)

        f.close()
#这个n-gram做的是以2 3 4  byte为单位，在数据集上扫描
