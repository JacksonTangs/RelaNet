#最后用这个处理抓到的网络流量包即可
import pandas as pd
import csv
import re
import numpy as np
import argparse
import os
def check_string(re_exp, str):
    res = re.search(re_exp, str)
    if res:
        return True
    else:
        return False
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default="", help='input file')
parser.add_argument('--output_file', type=str, default="", help='output file')
args = parser.parse_args()


#'/content/drive/MyDrive/Colab Notebooks/data/modbus-top-6000.txt'

#filename = '/content/drive/MyDrive/Colab Notebooks/data/01-21-top-1500.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
filename = args.input_file # txt文件和当前脚本在同一目录下，所以不用写具体路径
outputfile_name = args.output_file
pos = []

Efield = []
line_count = 1
os.chdir('/content/drive/My Drive/Colab Notebooks/data')
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行读取数据
        #print(lines)
        if not lines:
            break
            pass

        if lines != '+---------+---------------+----------+\n'and lines != '\n':
            if not check_string('ETHER$', lines):
#            print(1)
                if line_count%1000 == 0:
                  print("dealing with "+str(line_count)+" line trace")
                tmp = lines.split('|') # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                tmp_new = np.array(tmp[2:])
                #pos.append(tmp_new[2:])  # 添加新读取的数据
                pos.append(tmp[2:])  # 添加新读取的数据
                #np.savetxt('dns_new.csv',tmp[2:],delimiter=',',fmt = '%s')
                #Efield.append(tmp)
                #print(tmp_new)
                line_count = line_count + 1
                if line_count > 800000:
                  break
    pos = np.array(pos) # 将数据从list类型转换为array类型。

    #Efield = np.array(Efield)
    pass
#print(pos)
#np.savetxt('dns_new.csv',pos,delimiter=',',fmt = '%s')
outputfile_name
#'/content/drive/MyDrive/Colab Notebooks/data/modbus-top-6000.csv'
with open(outputfile_name,mode='w',newline='',encoding='utf8') as cf:
    wf=csv.writer(cf)
    for i in range(len(pos)):
        for j in range(len(pos[i])):
            #print(pos[i][j])
            if pos[i][j]!='\n':
                pos[i][j] = '0x'+pos[i][j]
                pos[i][j] = int(pos[i][j],16)

    line_count = 1
    for i in pos:
        wf.writerow(i)
        print('line '+str(line_count)+'  : Finish  !')
        line_count = line_count + 1
print(pos)