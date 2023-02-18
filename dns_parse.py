def deal_domain_name(offset_pointer,row,type_flag_array):
    if int(row[offset_pointer]) != 0:
        for i in range(0,int(row[offset_pointer])+1):
            type_flag_array.append(1)#添加dns域名为可变字段
        offset_pointer =  offset_pointer + int(row[offset_pointer]) + 1
        return deal_domain_name(offset_pointer,row,type_flag_array)
    else :
        return offset_pointer,type_flag_array

def parse_dns_test(row):
    offset_pointer = 0
    type_flag_array = []
    extract_model = []
    trans_id = tuple(row[0:2])
    for i in range(0,2):#Transaction ID belong to no-fix field
        type_flag_array.append(1)

    flags = tuple(row[2:4])
    for i in range(0,2):#Flags belong to fix field
        type_flag_array.append(2)
    offset_msg = 2
    extract_model.append((flags,offset_msg))
    num_ques = int(row[4])*256 + int(row[5])
    for i in range(0,2):#Num question belong to fix field
        type_flag_array.append(2)


    num_ans_RR = int(row[6])*256 + int(row[7])
    if flags[0] == 1 and flags[1] == 0:#如果这个包的类型属于询问类型
        for i in range(0,2):#Num RR belong to fix field
            type_flag_array.append(2)
    else:
        type_flag_array.append(2)
        type_flag_array.append(1)

    num_auth_RR = int(row[8])*256 + int(row[9])
    if flags[0] == 1 and flags[1] == 0:#如果这个包的类型属于询问类型
        for i in range(0,2):#Num RR belong to fix field
            type_flag_array.append(2)
    else:
        type_flag_array.append(2)
        type_flag_array.append(1)



    num_addi_RR = int(row[10])*256 + int(row[11])
    if flags[0] == 1 and flags[1] == 0:#如果这个包的类型属于询问类型
        for i in range(0,2):#Num RR belong to fix field
            type_flag_array.append(2)
    else:
        type_flag_array.append(2)
        type_flag_array.append(1)


    #type_flag_array.append(1)#这个是因为长度类型字段是个可变字段
    offset_pointer = 12
    #下面部分是处理query字段内容
    try :
        for i in range(0,num_ques):
            offset_pointer,type_flag_array = deal_domain_name(offset_pointer,row,type_flag_array)
            #print('offset_pointer',offset_pointer)
            offset_pointer = offset_pointer + 1#将指针向右移一格放到type类型上
            type_flag_array.append(1)#DNS域名字段结束时末尾追加
            temp_type_query = tuple(row [ offset_pointer : offset_pointer + 2])
            extract_model.append((temp_type_query,offset_pointer))

            offset_pointer = offset_pointer + 4
            for j in range(0,2):#TYPE 是不固定字段
                type_flag_array.append(1)
            for j in range(0,2):#CLASS 是固定字段
                type_flag_array.append(2)
    except IndexError:
        print("packet too long !!!!")
    print("在处理完query后type_flag_array的长度是： ",len(type_flag_array))
    #下面部分是处理response部分answer处理
    #try :
    for i in range(0,num_ans_RR):

        #offset_pointer = offset_pointer + 2#要略过response开始的NAME字段
        #print("row[offset_pointer] : ",row[offset_pointer])
        #print("row[offset_pointer+1] : ",row[offset_pointer+1])
        #print(type)
        print("ANSWER_NAME : ",row[offset_pointer:offset_pointer+2])
        if int(row[offset_pointer]) == 192 and int(row[offset_pointer+1]) == 12:
            #print("********************* 192 12 done***********************")
            type_flag_array.append(1)#CO 12
            type_flag_array.append(2)
            offset_pointer = offset_pointer + 2#要略过response开始的NAME字段
            temp_type_response = tuple(row[offset_pointer:offset_pointer+2])
            extract_model.append((temp_type_response,offset_pointer))#将response的元组添加到抽象模型中
            type_flag_array.append(2)#TYPE
            type_flag_array.append(1)

        else :

            if int(row[offset_pointer]) >= 192 and int(row[offset_pointer]) <= 255:
                offset_pointer = offset_pointer + 2#要略过response开始的NAME字段
                for j in range(0,2):#C0 XX 字段给blue
                    type_flag_array.append(1)
                temp_type_response = tuple(row[offset_pointer:offset_pointer+2])
                extract_model.append((temp_type_response,offset_pointer))#将response的元组添加到抽象模型中

                for j in range(0,2):#TYPE
                    type_flag_array.append(1)

            else :

                for j in range(0,int(row[offset_pointer]) + 2 +1):#C0 XX 字段暂时给blue
                    type_flag_array.append(1)
                offset_pointer = offset_pointer + int(row[offset_pointer]) + 2 + 1


                temp_type_response = tuple(row[offset_pointer:offset_pointer+2])
                extract_model.append((temp_type_response,offset_pointer))#将response的元组添加到抽象模型中

                for j in range(0,2):#TYPE
                    type_flag_array.append(1)
        for j in range(0,2):#CLASS
            type_flag_array.append(2)
        offset_pointer = offset_pointer + 2 + 2 #略过 CLASS 字段


        #print("row[offset_pointer] : ",row[offset_pointer])
        #print("row[offset_pointer+1] : ",row[offset_pointer+1])
        #print("row[offset_pointer+2] : ",row[offset_pointer+2])
        #print("row[offset_pointer+3] : ",row[offset_pointer+3])
        if int(row[offset_pointer]) == 0 and int(row[offset_pointer+1]) == 0\
            and int(row[offset_pointer+2]) == 2 and int(row[offset_pointer+3]) == 88 :
            for j in range(0,3):#TTL
                type_flag_array.append(1)
            for j in range(0,1):#TTL
                type_flag_array.append(2)
            offset_pointer = offset_pointer + 4 #略过 TTL 字段

            type_flag_array.append(2)#length
            type_flag_array.append(1)
        else :
            for j in range(0,4):#TTL
                type_flag_array.append(1)

            offset_pointer = offset_pointer + 4 #略过 TTL 字段
            for j in range(0,2):#length
                type_flag_array.append(1)



        rddata_length = int(row[offset_pointer])*256 + int(row[offset_pointer+1])
        offset_pointer = offset_pointer +2 #先略过length字段

        offset_pointer = offset_pointer + rddata_length
        for j in range(0,rddata_length):#length包括的字段为blue
            type_flag_array.append(1)
    #except IndexError:
        #print("packet too long !!!!")
    print("在处理完answer后type_flag_array的长度是： ",len(type_flag_array))
    #下面部分是处理response部分authority处理
    try :
        for i in range(0,num_auth_RR):
            if row[offset_pointer] == '0':
                offset_pointer = offset_pointer + 1
                type_flag_array.append(1)
            else :
                if int(row[offset_pointer]) == 192 and int(row[offset_pointer+1]) == 12 :
                    #print("********************* 192 12 done***********************")
                    type_flag_array.append(1)#CO 12
                    type_flag_array.append(2)
                    offset_pointer = offset_pointer + 2#要略过response开始的NAME字段
                    temp_type_response = tuple(row[offset_pointer:offset_pointer+2])
                    extract_model.append((temp_type_response,offset_pointer))#将response的元组添加到抽象模型中
                    type_flag_array.append(2)#TYPE
                    type_flag_array.append(1)
                else:

                    if int(row[offset_pointer]) >= 192 and int(row[offset_pointer]) <= 255:
                        offset_pointer = offset_pointer + 2#要略过response开始的NAME字段
                        for j in range(0,2):#C0 XX 字段暂时给blue
                            type_flag_array.append(1)
                        temp_type_response = tuple(row[offset_pointer:offset_pointer+2])
                        extract_model.append((temp_type_response,offset_pointer))#将response的元组添加到抽象模型中

                        for j in range(0,2):#TYPE
                            type_flag_array.append(1)

                    else :

                        for j in range(0,int(row[offset_pointer]) + 2 +1):#C0 XX 字段暂时给blue
                            type_flag_array.append(1)
                        offset_pointer = offset_pointer + int(row[offset_pointer]) + 2 + 1


                        temp_type_response = tuple(row[offset_pointer:offset_pointer+2])
                        extract_model.append((temp_type_response,offset_pointer))#将response的元组添加到抽象模型中

                        for j in range(0,2):#TYPE
                            type_flag_array.append(1)
            for j in range(0,2):#CLASS
                type_flag_array.append(2)
            offset_pointer = offset_pointer + 2 + 2 #略过 CLASS 字段
            #print("row[offset_pointer] : ",row[offset_pointer])
            #print("row[offset_pointer+1] : ",row[offset_pointer+1])
            #print("row[offset_pointer+2] : ",row[offset_pointer+2])
            #print("row[offset_pointer+3] : ",row[offset_pointer+3])
            if int(row[offset_pointer]) == 0 and int(row[offset_pointer+1]) == 0\
                and int(row[offset_pointer+2]) == 2 and int(row[offset_pointer+3]) == 88 :
                for j in range(0,3):#TTL
                    type_flag_array.append(1)
                for j in range(0,1):#TTL
                    type_flag_array.append(2)
                offset_pointer = offset_pointer + 4 #略过 TTL 字段

                type_flag_array.append(2)#length
                type_flag_array.append(1)
            else :
                for j in range(0,4):#TTL
                    type_flag_array.append(1)

                offset_pointer = offset_pointer + 4 #略过 TTL 字段
                for j in range(0,2):#length
                    type_flag_array.append(1)


            rddata_length = int(row[offset_pointer])*256 + int(row[offset_pointer+1])
            offset_pointer = offset_pointer +2 #先略过length字段

            offset_pointer = offset_pointer + rddata_length
            for j in range(0,rddata_length):#length包括的字段为blue
                type_flag_array.append(1)
    except IndexError:
        print("packet too long !!!!")
    print("在处理完authority后type_flag_array的长度是： ",len(type_flag_array))
    #下面部分是处理response部分additional处理
    try :
        for i in range(0,num_addi_RR):
            if row[offset_pointer] == '0':
                offset_pointer = offset_pointer + 1
                type_flag_array.append(1)
                temp_type_response = tuple(row[offset_pointer:offset_pointer+2])
                extract_model.append((temp_type_response,offset_pointer))#将response的元组添加到抽象模型中


                for j in range(0,2):#TYPE
                    type_flag_array.append(1)
                for j in range(0,2):#CLASS
                    type_flag_array.append(2)

                for j in range(0,4):#TTL
                    type_flag_array.append(1)
                offset_pointer = offset_pointer + 2 + 2 + 1 + 1 + 2  #略过一堆


                rddata_length = int(row[offset_pointer])*256 + int(row[offset_pointer+1])
                offset_pointer = offset_pointer +2 #先略过length字段
                for j in range(0,2):#length
                    type_flag_array.append(1)
                offset_pointer = offset_pointer + rddata_length
                for j in range(0,rddata_length):#length包括的字段为blue
                    type_flag_array.append(1)


            else :
                #print("NAME: ",row[offset_pointer:offset_pointer+2])
                #print("row[offset_pointer+1] : ",row[offset_pointer+1])
                offset_pointer = offset_pointer + 2#要略过response开始的NAME字段
                for j in range(0,2):#要略过response开始的NAME字段
                    type_flag_array.append(1)

                temp_type_response = tuple(row[offset_pointer:offset_pointer+2])
                extract_model.append((temp_type_response,offset_pointer))#将response的元组添加到抽象模型中

                #print("CLASS +TYPE: ",row[offset_pointer:offset_pointer+4])
                for j in range(0,2):#TYPE
                    type_flag_array.append(1)
                for j in range(0,2):#CLASS
                    type_flag_array.append(2)
                offset_pointer = offset_pointer + 2 + 2 #略过 CLASS 字段
                #print("row[offset_pointer] : ",row[offset_pointer])
                #print("row[offset_pointer+1] : ",row[offset_pointer+1])
                #print("row[offset_pointer+2] : ",row[offset_pointer+2])
                #print("row[offset_pointer+3] : ",row[offset_pointer+3])
                if int(row[offset_pointer]) == 0 and int(row[offset_pointer+1]) == 0\
                    and int(row[offset_pointer+2]) == 2 and int(row[offset_pointer+3]) == 88 :
                    for j in range(0,3):#TTL
                        type_flag_array.append(1)
                    for j in range(0,1):#TTL
                        type_flag_array.append(2)
                    offset_pointer = offset_pointer + 4 #略过 TTL 字段

                    type_flag_array.append(2)#length
                    type_flag_array.append(1)

                else :
                    for j in range(0,4):#TTL
                        type_flag_array.append(1)


                    offset_pointer = offset_pointer + 4 #略过 TTL 字段
                    for j in range(0,2):#length
                        type_flag_array.append(1)


                rddata_length = int(row[offset_pointer])*256 + int(row[offset_pointer+1])
                offset_pointer = offset_pointer +2 #先略过length字段

                offset_pointer = offset_pointer + rddata_length
                #print("rddata_length length is: ",rddata_length)
                for j in range(0,rddata_length):#length包括的字段为blue
                    type_flag_array.append(1)
    except IndexError:
        print("packet too long !!!!")


    print("在处理完additional后type_flag_array的长度是： ",len(type_flag_array))
    print('one packet test making finish !!!!!!!!!!!! ')
    #print(extract_model)
    #print('up is extract model!!!!')
    #print(type_flag_array)
    #print('up is MARKED model!!!!')
    return extract_model,type_flag_array