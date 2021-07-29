import sys
import re

rst = {}
header = ""

with open(sys.argv[1]) as f:
    for line in f:
        if line.startswith("Linux"):
            continue

        if line.startswith("#"):
            if not header:
                header = line
            continue
        line = re.sub(' +', ' ', line).split()

        if len(line) == 0:
            continue

        for i in range(-16, -1):
            if i == -11:
                continue
            if not i in rst:
                rst[i] = line
            else:
                if float(line[i]) > float(rst[i][i]):
                    rst[i] = line

header = re.sub(' +', ' ', header).split()

for k, v in rst.items():
    if (header[k] == "VSZ" or header[k] == "RSS"):
        print("max", header[k], round(float(v[k])/1024/1024, 2), "GB")
    else:
        print("max", header[k], v[k])

    # print(v)
    # print("==========")