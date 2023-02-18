def deal_domain_name(offset_pointer,row,type_flag_array):
    if int(row[offset_pointer]) != 0:
        for i in range(0,int(row[offset_pointer])+1):
            type_flag_array.append(1)#添加dns域名为可变字段
        offset_pointer =  offset_pointer + int(row[offset_pointer]) + 1
        return deal_domain_name(offset_pointer,row,type_flag_array)
    else :
        return offset_pointer,type_flag_array

def parse_dhcp_test(row):
    offset_pointer = 0
    type_flag_array = []
    extract_model = []


    type_flag_array.append(2)#message_type
    flags = tuple(row[0:1])
    offset_pointer = offset_pointer+1


    type_flag_array.append(2)#hardware type
    peer_clock = tuple(row[1:2])
    offset_pointer = offset_pointer+1

    type_flag_array.append(2)#hardware length
    pollying_interval = tuple(row[2:3])
    offset_pointer = offset_pointer+1

    type_flag_array.append(2)#hops
    clock_precision = tuple(row[3:4])
    offset_pointer = offset_pointer+1

    for i in range(0,4):#transaction id
        type_flag_array.append(1)
    root_delay = tuple(row[4:8])
    offset_pointer = offset_pointer+4

    for i in range(0,2):#seconds elapsed
        type_flag_array.append(1)
    root_dispersion = tuple(row[offset_pointer:offset_pointer+2])
    offset_pointer = offset_pointer+2

    #bootp flags
    if list(map(int,row[offset_pointer:offset_pointer+2]))==[0,0]:
        type_flag_array.append(2)
        type_flag_array.append(1)
    elif list(map(int,row[offset_pointer:offset_pointer+2]))==[128,0]:
        type_flag_array.append(1)
        type_flag_array.append(2)
    else:
        type_flag_array.append(1)
        type_flag_array.append(1)

    offset_pointer = offset_pointer+2

    #client ip


    for i in range(0,4):#client ip
        type_flag_array.append(1)
    offset_pointer = offset_pointer+4

    for i in range(0,10):#clienr hardware address padding
        type_flag_array.append(2)
    #transmit_timestamp = tuple(row[offset_pointer:offset_pointer+10])
    offset_pointer = offset_pointer+10

    for i in range(0,64):#server host name
        type_flag_array.append(2)
    #transmit_timestamp = tuple(row[offset_pointer:offset_pointer+64])
    offset_pointer = offset_pointer+64

    for i in range(0,128):#boot file name
        type_flag_array.append(1)
    #transmit_timestamp = tuple(row[offset_pointer:offset_pointer+128])
    offset_pointer = offset_pointer+128

    for i in range(0,4):#magic cookie
        type_flag_array.append(1)
    #transmit_timestamp = tuple(row[offset_pointer:offset_pointer+4])
    offset_pointer = offset_pointer+4
    print('option: ',int(row[offset_pointer]))
    try:
        while int(row[offset_pointer]) !=255:


            if offset_pointer+int(row[offset_pointer+1])+2 > len(row):
                for i in range(0,len(row)-offset_pointer):
                    type_flag_array.append(1)
                break
            else:
                for i in range(0,int(row[offset_pointer+1])+1):
                    type_flag_array.append(1)

                #print('option length: ',int(row[offset_pointer+1])+1)
                #print('row:',row)
                #print(row[offset_pointer:offset_pointer+int(row[offset_pointer+1])+2])

                offset_pointer = offset_pointer+int(row[offset_pointer+1])+2



                #print(hex(int(row[offset_pointer-4])),hex(int(row[offset_pointer-3])),hex(int(row[offset_pointer-2])),)
                #print('option: ',int(row[offset_pointer]))
                #offset_pointer,type_flag_array = deal_domain_name(offset_pointer,row,type_flag_array)

    except IndexError:
        if len(row) == 400:
            for i  in range(0,400-1-offset_pointer):
                type_flag_array.append(1)
    finally:
        print('len(type_flag_array)',len(type_flag_array))
        return [],type_flag_array