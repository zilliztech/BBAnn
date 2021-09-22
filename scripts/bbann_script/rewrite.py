#!/usr/bin/python3
import sys

column_num=eval(sys.argv[1])
print("ARGUMENT column_num: ", column_num)

file_name = "tana_res.txt"
records = {}
with open(file_name) as f:
	while True:
		line = f.readline() # Line 1: log file name
		if not line:
			break
		print(line.strip())
		key = f.readline().strip() # Line 2: the key == nprobe, refine_nprobe
		print("key: ${0}".format(key))
		new_record = []
		for x in range(column_num):  # Line 3-n
			new_record.append(f.readline().strip())
		records[key] = new_record

sorted_records = sorted(records.items())

outfile="form.txt"
with open(outfile, 'w') as f:
	for kv in sorted_records:
		f.write(kv[0] + " ")
		for val in kv[1]:
			f.write(val)
			f.write(' ')
		f.write('\n')
