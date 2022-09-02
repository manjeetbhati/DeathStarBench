import os

# file_name = ["social_network_ComposePostService.lua", "social_network_ttypes.lua", "social_network_constants.lua"]
file_name = ["social_network_MediaService.lua"]

for file in file_name:
    f = open(file, 'r')
    f1 = open("tmp.lua", 'w')
    lines = f.readlines()
    lines[-1] = lines[-1] + "\n"
    flag = False
    count = 0
    count1 = 0
    upper_lines = []
    lower_lines = []
    package_lines = []
    package_index = 0
    upper_flag = True
    for line in lines:
        if line[0] == "\n":
            if upper_flag:
                upper_lines.append(line[:-1])
            else:
                lower_lines.append(line[:-1])
            # print >> f1, line[:-1]
            continue
        if line[:2] == "--":
            # print >> f1, line[:-1]
            if line == "-- HELPER FUNCTIONS AND STRUCTURES\n":
                upper_flag = False
            continue

        if line[:7] == "require":
            package_name = line[9:-2]
            package_lines.append("local " + package_name + " = " + line[:-1])
            if lines.index(line) > package_index:
                package_index = lines.index(line)
            continue
        if flag:
            if line[:3] == "end":
                flag = False
            if upper_flag:
                upper_lines.append(line[:-1])
            else:
                lower_lines.append(line[:-1])
            # print >> f1, line[:-1]
            continue
        if line[:8] != "function":
            if count == 0 and count1 == 0:
                if upper_flag:
                    upper_lines.append("local " + line[:-1])
                else:
                    lower_lines.append("local " + line[:-1])
                # print >> f1, "local " + line[:-1]
                count += line.count("{") - line.count("}")
                count1 += line.count("(") - line.count(")")
            else:
                if upper_flag:
                    upper_lines.append(line[:-1])
                else:
                    lower_lines.append(line[:-1])
                # print >> f1, line[:-1]
                count += line.count("{") - line.count("}")
                count1 += line.count("(") - line.count(")")
        else:
            flag = True
            if upper_flag:
                upper_lines.append(line[:-1])
            else:
                lower_lines.append(line[:-1])
            # print >> f1, line[:-1]
            count += line.count("{") - line.count("}")
            count1 += line.count("(") - line.count(")")

    if file_name.index(file) == 0:
        package_lines.append("local TType = Thrift.TType")
        package_lines.append("local TMessageType = Thrift.TMessageType")
        package_lines.append("local __TObject = Thrift.__TObject")
        package_lines.append("local TApplicationException = Thrift.TApplicationException")
        package_lines.append("local __TClient = Thrift.__TClient")
        package_lines.append("local __TProcessor = Thrift.__TProcessor")
        package_lines.append("local ttype = Thrift.ttype")
        package_lines.append("local ttable_size = Thrift.ttable_size")
        package_lines.append("local ServiceException = auto_microservices_ttypes.ServiceException")
    if file_name.index(file) == 1:
        package_lines.append("local TType = Thrift.TType")
        package_lines.append("local __TObject = Thrift.__TObject")
        package_lines.append("local TException = Thrift.TException")
    for i in package_lines:
        print >> f1, i
    for i in lower_lines:
        print >> f1, i
    for i in upper_lines:
        print >> f1, i
    if package_index + 2 < len(lines):
        return_client_line = lines[package_index + 2]
        return_client = return_client_line[:(return_client_line.index(" "))]
        if file_name.index(file) == 0:
            print >> f1, "return " + return_client

    if file_name.index(file) == 1:
        print >> f1, "return {ErrorCode=ErrorCode, ServiceException=ServiceException}"
    f.close()
    f1.close()
    os.popen('mv tmp.lua ' + file)
