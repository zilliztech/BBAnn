import sys
import re

rst = {}
header = ""

def split_to_list(l):
    return re.sub(' +', ' ', l).split()

with open(sys.argv[1]) as f:
    for line in f:
        if line.startswith("Linux"):
            continue

        if line.startswith("#"):
            if not header:
                header = split_to_list(line)
            continue
        line = split_to_list(line)

        if len(line) == 0:
            continue

        i = -2
        while True:
            print(header[i], i)
            if header[i] == "CPU":
                i -= 1
                continue
            if not i in rst:
                rst[i] = line
            else:
                if float(line[i]) > float(rst[i][i]):
                    rst[i] = line

            if header[i] == "%usr":
                break
            
            i -= 1

for k, v in rst.items():
    if (header[k] == "VSZ" or header[k] == "RSS"):
        print("max", header[k], round(float(v[k])/1024/1024, 2), "GB")
    else:
        print("max", header[k], v[k])

    # print(v)
    # print("==========")