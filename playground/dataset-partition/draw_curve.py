import json
import matplotlib.pyplot as plt


def get_cluster_nn_classification(json_dir):
    x_arr = []
    y_arr = []
    with open(json_dir, 'r') as file:
        json_data = json.load(file)
        for ele in json_data:
            if ele['recall'] == 0.0:
                continue
            x_arr.append(ele['n_candidate'])
            y_arr.append(ele['recall'])
    return x_arr, y_arr


# deep gist glove imagenet sift normalsmall
dataset_name = 'sift1M'

method_arr = [
    'graph_partition-knn-k_30', 'kmeans'
]
dir_arr = []
for i in range(len(method_arr)):
    tmp_str = "./result/%s-%s/item_recall_curve.json" % (dataset_name, method_arr[i])
    dir_arr.append(tmp_str)

cls_arr = []
for i in range(len(dir_arr)):
    cls_tmp = get_cluster_nn_classification(dir_arr[i])
    cls_arr.append(cls_tmp)

# 第一个是横坐标的值，第二个是纵坐标的值
plt.figure(num=3, figsize=(8, 5))
# marker
# o 圆圈, v 倒三角, ^ 正三角, < 左三角, > 右三角, 8 大圆圈, s 正方形, p 圆圈, * 星号, h 菱形, H 六面体, D 大菱形, d 瘦的菱形, P 加号, X 乘号
# 紫色#b9529f 蓝色#3953a4 红色#ed2024 #231f20 深绿色#098140 浅绿色#7f8133 #0084ff
# solid dotted
marker_l = ['H', 'D', 'P', '>', '*', 'X', 's', '<', '^', 'p', 'v']
color_l = ['#b9529f', '#3953a4', '#ed2024', '#098140', '#231f20', '#7f8133', '#0084ff']

for i in range(len(dir_arr)):
    line = plt.plot(cls_arr[i][0], cls_arr[i][1], marker=marker_l[i], linestyle='solid', color=color_l[i],
                    label=method_arr[i])
# line2_1, = plt.plot(cls_arr[0][0], cls_arr[0][1], marker='H', linestyle='solid', color='#b9529f',
#                     label=method_arr[0])
# line2_2, = plt.plot(cls_arr[1][0], cls_arr[1][1], marker='D', linestyle='solid', color='#3953a4',
#                     label=method_arr[1])
# line2_3, = plt.plot(cls_arr[2][0], cls_arr[2][1], marker='P', linestyle='solid', color='#ed2024',
#                     label=method_arr[2])
# line2_5, = plt.plot(cls_arr[3][0], cls_arr[3][1], marker='>', linestyle='solid', color='#098140',
#                     label=method_arr[3])
# line2_6, = plt.plot(cls_arr[4][0], cls_arr[4][1], marker='*', linestyle='solid', color='#231f20',
#                     label=method_arr[4])
# line2_7, = plt.plot(cls_arr[5][0], cls_arr[5][1], marker='P', linestyle='solid', color='#ed2024',
#                     label='4 partition knn')

plt.xscale('log')
# plt.xlim(1, 500000)

# line, = plt.plot(curve[0], curve[1], marker='o', linestyle='solid', label='$M$: 2', color='#b9529f')

# 使用ｌｅｇｅｎｄ绘制多条曲线
# plt.title('graph kmeans vs knn')
plt.legend(loc='upper left', title="%s 1M, top-10" % dataset_name)

plt.xlabel("the number of candidates")
plt.ylabel("Recall")
plt.grid(True, linestyle='-.')
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
# plt.yticks([0.75, 0.8, 0.85])
plt.savefig('item-recall-curve.png')
plt.close()
