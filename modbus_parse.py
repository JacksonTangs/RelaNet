def deal_domain_name(offset_pointer,row,type_flag_array):
    if int(row[offset_pointer]) != 0:
        for i in range(0,int(row[offset_pointer])+1):
            type_flag_array.append(1)#添加dns域名为可变字段
        offset_pointer =  offset_pointer + int(row[offset_pointer]) + 1
        return deal_domain_name(offset_pointer,row,type_flag_array)
    else :
        return offset_pointer,type_flag_array

def parse_modbus_test(row):
    offset_pointer = 0
    type_flag_array = []
    extract_model = []
    trans_id = tuple(row[0:2])
    for i in range(0,2):#Transaction ID belong to no-fix field
        type_flag_array.append(1)

    protocol_id = tuple(row[2:4])
    for i in range(0,2):#Protocol ID
        type_flag_array.append(2)
    offset_msg = 2
    extract_model.append((protocol_id,offset_msg))

    Length = int(row[4])*256 + int(row[5])
    for i in range(0,2):#Length 不是6就是5 29
        type_flag_array.append(2)

    for i in range(0,1):#Unit Identifier
        type_flag_array.append(2)
    unit_id = int(row[6])*256 + int(row[7])

    for i in range(0,1):#Function Code
        type_flag_array.append(2)



    #一种情况是reference number + word count
    #还一种是 byte count+
    if Length == 6 or Length == 5:
        for i in range(0,Length-2):
            type_flag_array.append(2)

    elif Length == 29:
        for i in range(0,1):#byte count
            type_flag_array.append(2)
        for i in range(0,Length-3):#All register value
            type_flag_array.append(2)
    else:
        print("Parse Modbus Wrong !!!")
    return [],type_flag_array
