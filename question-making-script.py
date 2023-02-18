import os
b = os.chdir('/content/drive/My Drive/Colab Notebooks')
#python main_test.py --resume 'epoch_RN_20.pth' --dic_rank 1000 --ans_prob_threshold 0.97 --fuzz_range 5
protocol_list = ['ntp','dnp3','modbus','dhcp']
protocol_list = ['ntp']
dic_rank_list =[1000,127,1400]

dic_rank_list = [1000]
fuzz_range_list = [5]
fuzz_range_list = [0,1,2,3,4,5,6,7,8,9,10]
#fuzz_range_list = [0]
ans_prob_threshold_list = [0.95,0.96,0.97,0.98,0.99]
protocol_type = 'dhcp'
protocol_type = 'dhcp'
for protocol_type in protocol_list:
    for dic_rank in dic_rank_list:
        for fuzz_range in fuzz_range_list:
            #dic_rank = 400 600 800 1000 1200
            #fuzz_range = 0 3 5 7 10 15 20
            #data_file_name = "04-17-dns-pure-"+str(file_count)+"00.pcap"
            #output_dir_name = "dns_"+str(file_count)+"00"
            #a = os.system(r"python /home/tangtong/netplier_pycharm/netplier/main.py -i data/"+data_file_name+" -o /home/tangtong/netplier_pycharm/tmp/"+output_dir_name+" -t dns > log_dns_"+str(file_count)+".txt")
            execute_cmd = "python question-making.py --protocol_type "+protocol_type+" --fuzz_range "+str(fuzz_range)+" --dictionay_choose "+str(dic_rank)
            #execute_cmd = "python question-making-diff.py --protocol_type "+protocol_type+" --fuzz_range "+str(fuzz_range)+" --dictionay_choose "+str(dic_rank)

            a = os.system(execute_cmd)