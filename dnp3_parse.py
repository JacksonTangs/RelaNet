def deal_domain_name(offset_pointer,row,type_flag_array):
    if int(row[offset_pointer]) != 0:
        for i in range(0,int(row[offset_pointer])+1):
            type_flag_array.append(1)#添加dns域名为可变字段
        offset_pointer =  offset_pointer + int(row[offset_pointer]) + 1
        return deal_domain_name(offset_pointer,row,type_flag_array)
    else :
        return offset_pointer,type_flag_array

def parse_dnp3_test(row):
    offset_pointer = 0
    type_flag_array = []
    extract_model = []

    for i in range(0,2):
        type_flag_array.append(2)#start bytes
    flags = tuple(row[0:1])
    offset_pointer = offset_pointer+2


    type_flag_array.append(2)#length
    length = tuple(row[1:2])
    offset_pointer = offset_pointer+1

    type_flag_array.append(2)#control
    control = tuple(row[offset_pointer:offset_pointer+1])
    offset_pointer = offset_pointer+1


    for i in range(0,2):#destination
        type_flag_array.append(2)
    root_delay = tuple(row[4:8])
    offset_pointer = offset_pointer+2

    for i in range(0,2):#source
        type_flag_array.append(2)
    root_dispersion = tuple(row[8:12])
    offset_pointer = offset_pointer+2

    for i in range(0,2):#data link header checksum
        type_flag_array.append(2)
    peer_clock = tuple(row[12:16])
    offset_pointer = offset_pointer+2

    #type_flag_array.append(1)#transport control
    #control = tuple(row[offset_pointer:offset_pointer+1])
    #offset_pointer = offset_pointer+1

    #下面开始处理data chunk部分
    #if offset_pointer+16+2 >len(row):
    #    break

    flag_1 = 0
    flag_2 = 0
    flag_3 = 0
    flag_3_sign = 0
    flag_4 = 0
    flag_4_sign = 0

    flag_5 = 0
    while int(row[offset_pointer+16])!=0 and int(row[offset_pointer+17])!=0:
        for i in range(0,16):
            type_flag_array.append(1)
        offset_pointer = offset_pointer +1



        type_flag_array.append(1)#
        type_flag_array.append(2)#



        offset_pointer = offset_pointer +2



        if offset_pointer+16+2 > len(row):
            for i in range(0,len(row)-offset_pointer):
                type_flag_array.append(1)#data trunk
                offset_pointer = offset_pointer+1
            break

    #接下来要处理的就是后面都为0的情况
    temp_offset_pointer = offset_pointer+16
    for j in range(0,16):
        if int(row[temp_offset_pointer])!=0 and int(row[temp_offset_pointer+1])!=0:
            for i in range(0,temp_offset_pointer-offset_pointer+2):
                if (int(row[offset_pointer-2]) == 129 or int(row[offset_pointer-2]) == 130) and\
                int(row[offset_pointer-1]) == 0 and int(row[offset_pointer]) == 0:
                    type_flag_array.append(2)
                elif offset_pointer == 13 and int(row[offset_pointer-1]) == 129 and int(row[offset_pointer]) == 0\
                and int(row[offset_pointer+1]) != 0:
                    type_flag_array.append(2)
                else:
                    type_flag_array.append(1)
                offset_pointer = offset_pointer +1


            break
        else:
            temp_offset_pointer = temp_offset_pointer -1


    '''

    if offset_pointer+16+2< len(row):

    else:
        for i in range(0,len(row)-offset_pointer):
            type_flag_array.append(1)#data trunk
            offset_pointer = offset_pointer+1
        break

    while offset_pointer+16+2 < len(row):
        for i in range(0,16):
            type_flag_array.append(1)#data trunk
        offset_pointer = offset_pointer+16
    for i in range(0,len(row)-offset_pointer):
        type_flag_array.append(1)#data trunk
        offset_pointer = offset_pointer+1
    '''
    return [],type_flag_array


