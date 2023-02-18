def deal_domain_name(offset_pointer,row,type_flag_array):
    if int(row[offset_pointer]) != 0:
        for i in range(0,int(row[offset_pointer])+1):
            type_flag_array.append(1)#添加dns域名为可变字段
        offset_pointer =  offset_pointer + int(row[offset_pointer]) + 1
        return deal_domain_name(offset_pointer,row,type_flag_array)
    else :
        return offset_pointer,type_flag_array

def parse_ntp_test(row):
    #FLAGS
    # 1
    offset_pointer = 0
    type_flag_array = []
    extract_model = []


    type_flag_array.append(2)#flags
    flags = tuple(row[0:1])
    offset_pointer = offset_pointer+1


    type_flag_array.append(2)#peer clocks
    peer_clock = tuple(row[1:2])
    offset_pointer = offset_pointer+1

    type_flag_array.append(2)#pollying interval
    pollying_interval = tuple(row[2:3])
    offset_pointer = offset_pointer+1


    type_flag_array.append(1)#clock precision
    clock_precision = tuple(row[3:4])
    offset_pointer = offset_pointer+1

    for i in range(0,2):#root delay 前两个是0
        type_flag_array.append(2)
    offset_pointer = offset_pointer+2
    
    for i in range(0,2):#root delay
        type_flag_array.append(2)

    root_delay = tuple(row[4:8])
    offset_pointer = offset_pointer+2


    for i in range(0,2):#root_dispersion
        type_flag_array.append(2)
    for i in range(0,2):#root_dispersion
        type_flag_array.append(1)
    root_dispersion = tuple(row[8:12])
    offset_pointer = offset_pointer+4


    #if int(row[offset_pointer])== 172 and int(row[offset_pointer+1])== 19\
    #    and int(row[offset_pointer+2])== 1:
    if True:
        for i in range(0,3):
            type_flag_array.append(1)#reference id
    else:
        for i in range(0,3):
            type_flag_array.append(1)#reference id


    peer_clock = tuple(row[12:16])
    offset_pointer = offset_pointer+3
    if (int(row[offset_pointer]) == 2 and int(row[offset_pointer+1]) == 210)\
    or (int(row[offset_pointer]) == 135 and int(row[offset_pointer+1]) == 210):
        type_flag_array.append(2)#reference id
        type_flag_array.append(2)#reference timestamp
    else:
        type_flag_array.append(1)#reference id
        type_flag_array.append(1)#reference timestamp
    offset_pointer = offset_pointer+2




    for i in range(0,3):
        type_flag_array.append(1)#

    peer_clock = tuple(row[16:24])
    offset_pointer = offset_pointer+3


    for i in range(0,4):
        type_flag_array.append(1)#

        
    offset_pointer = offset_pointer+4






    type_flag_array.append(1)
    offset_pointer = offset_pointer+1
    for i in range(0,7):
        type_flag_array.append(1)#origin timestamp
    peer_clock = tuple(row[24:32])
    offset_pointer = offset_pointer+7

    for i in range(0,8):
        type_flag_array.append(1)#receive timestamp
    peer_clock = tuple(row[32:40])
    offset_pointer = offset_pointer+8

    for i in range(0,8):
        type_flag_array.append(1)#transmit timestamp
    transmit_timestamp = tuple(row[40:48])
    offset_pointer = offset_pointer+8

    if len(row) == offset_pointer :
        return [],type_flag_array
    else:
        for i in range(0,4):
            type_flag_array.append(1)#key_id
        #key_id = tuple(row[offset:offset+4])
        offset_pointer = offset_pointer+4

        for i in range(0,16):
            type_flag_array.append(1)#authentication_code
        #authentication_code = tuple(row[offset:offset+16])
        offset_pointer = offset_pointer+16
        return [],type_flag_array



