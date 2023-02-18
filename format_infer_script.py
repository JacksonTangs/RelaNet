
import os
b = os.chdir('/content/drive/My Drive/Colab Notebooks')
#python main_test.py --resume 'epoch_RN_20.pth' --dic_rank 1000 --ans_prob_threshold 0.97 --fuzz_range 5
#dic_rank_list =[400, 600 ,800 ,1000, 1200,1400]
#dic_rank_list =[400, 600 ,800 ,1000, 1200,1400]
#fuzz_range_list = [0 ,3, 5 ,7 ,10 ,15 ,20]
fuzz_range_list = [0,1,2,3,4,5,6,7,8,9,10]
#fuzz_range_list = [0]
ans_prob_threshold_list = [0.95,0.96,0.97,0.98,0.99]
ans_prob_threshold_list = [0.97]
dic_rank_list =[1068, 127, 386, 661, 1352]
protocol = "'ntp'"
protocol_type = 'ntp'
#fuzz_range_list = [0]

metric_threshold_list = [16,24,32,40,48,56,64,72,80,88,96,104,112,120,128]
for dic_rank in dic_rank_list:
    for fuzz_range in fuzz_range_list:
        for ans_prob_threshold in ans_prob_threshold_list:
            category = protocol_type+'-'+str(dic_rank)+'-fuzz-'+str(fuzz_range)+'-ans_prob_threshold-'+str(ans_prob_threshold)
            model_name  = 'softmax-result/'+category+'-inference_input.pickle'
            if os.path.exists(model_name):
                execute_cmd = r"python inference_cluster.py --fuzz "+str(fuzz_range)+" --dict "+str(dic_rank)+" --threshold "+str(ans_prob_threshold)+" --protocol "+protocol
                a = os.system(execute_cmd)
                for metric_threshold in metric_threshold_list:
                #dic_rank = 400 600 800 1000 1200
                #fuzz_range = 0 3 5 7 10 15 20
                #data_file_name = "04-17-dns-pure-"+str(file_count)+"00.pcap"
                #output_dir_name = "dns_"+str(file_count)+"00"
                #a = os.system(r"python /home/tangtong/netplier_pycharm/netplier/main.py -i data/"+data_file_name+" -o /home/tangtong/netplier_pycharm/tmp/"+output_dir_name+" -t dns > log_dns_"+str(file_count)+".txt")
                #execute_cmd = r"python main_test.py --resume 'epoch_RN_dict-"+str(dic_rank)+"-fuzz-"+str(fuzz_range)+".pth' --dic_rank "+str(dic_rank)+" --ans_prob_threshold "+str(ans_prob_threshold)+" --fuzz_range "+str(fuzz_range)
                #a = os.system(execute_cmd)
                    execute_cmd = r"python formats_inference.py --fuzz "+str(fuzz_range)+" --dict "+str(dic_rank)+" --threshold "+str(ans_prob_threshold)+" --protocol "+protocol+" --metric_threshold "+str(metric_threshold)
                    a = os.system(execute_cmd)