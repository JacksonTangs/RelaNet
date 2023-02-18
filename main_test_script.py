import os
b = os.chdir('/content/drive/My Drive/Colab Notebooks')
#python main_test.py --resume 'epoch_RN_20.pth' --dic_rank 1000 --ans_prob_threshold 0.97 --fuzz_range 5
protocol_list = ['ntp','dnp3','modbus','dhcp']
protocol_list = ['ntp']
dic_rank_list =[1000,127,1400]
dic_rank_list = [1000]
fuzz_range_list = [0,1,2,3,4,5,6,7,8,9,10]
#fuzz_range_list = [10]
ans_prob_threshold_list = [0.95,0.96,0.97,0.98,0.99]
ans_prob_threshold_list = [0.97]
protocol_type = 'dns'
protocol_type = 'ntp'
for protocol_type in protocol_list:
    for dic_rank in dic_rank_list:
        for fuzz_range in fuzz_range_list:
            for ans_prob_threshold in ans_prob_threshold_list:
                #dic_rank = 400 600 800 1000 1200
                #fuzz_range = 0 3 5 7 10 15 20
                #data_file_name = "04-17-dns-pure-"+str(file_count)+"00.pcap"
                #output_dir_name = "dns_"+str(file_count)+"00"
                #a = os.system(r"python /home/tangtong/netplier_pycharm/netplier/main.py -i data/"+data_file_name+" -o /home/tangtong/netplier_pycharm/tmp/"+output_dir_name+" -t dns > log_dns_"+str(file_count)+".txt")
                dirs='model/'
                model_name  = protocol_type+'-'+str(dic_rank)+'-fuzz-'+str(fuzz_range)+'epoch_RN_20.pth'
                if os.path.exists(dirs+model_name):
                    #execute_cmd = r"python main_test.py --resume "'epoch_RN_dict-"+str(dic_rank)+"-fuzz-"+str(fuzz_range)+".pth'" --dic_rank "+str(dic_rank)+" --ans_prob_threshold "+str(ans_prob_threshold)+" --fuzz_range "+str(fuzz_range)
                    execute_cmd = "python main_test.py --resume "+model_name+" --dic_rank "+str(dic_rank)+" --ans_prob_threshold "+str(ans_prob_threshold)+" --fuzz_range "+str(fuzz_range)+" --protocol_type "+protocol_type
                    a = os.system(execute_cmd)