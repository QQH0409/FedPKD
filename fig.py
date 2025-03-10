# 2023/09/14 22:39
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams.update({'font.size': 14})
#matplotlib.rcParams['font.family'] = 'Times New Roman'
# 定义 Y 轴的数据
x = ['LAN', '100Mbps', '10Mbps', '1Mbps']

y1 = [3438, 28686, 47641, 998114]
y2 = [150, 460, 1860, 3551]
y3 = [963, 8560, 14859, 123277]
y4 = [160, 489, 6235, 5920]
y5 = [42, 200, 1650, 14550]


# 自动生成 X 轴的坐标
#x = range(1, len(y) + 1)
#diff = [y_2[i] - y_1[i] for i in range(len(y_1))]
diff = np.array(y1) - np.array(y2)

#plt.figure(figsize=(6, 6))
plt.figure(1, figsize=(5, 5))

# 绘制折线图
bar_width = 0.35
#plt.vlines(x, np.minimum(y1, y2), np.maximum(y1, y2), colors='red', linestyles='dashed', zorder=3)   #体现差异
plt.plot(x, y1,marker='s',  linestyle='-', color=(31/255, 119/255, 180/255), label='Meadows', zorder=2)
plt.plot(x, y2, marker='^', linestyle='-', color=(255/255, 127/255, 14/255), label='kolesnikov', zorder=2)
plt.plot(x, y3,marker='s',  linestyle='-', color=(250/255, 190/255, 110/255), label='PPSSI', zorder=2)
plt.plot(x, y4, marker='^', linestyle='-', color=(155/255, 27/255, 194/255), label='Pinkas', zorder=2)
plt.plot(x, y5,marker='s',  linestyle='-', color=(91/255, 19/255, 18/255), label='ours', zorder=2)

#plt.bar(xx, y1, bar_width, color=(31/255, 119/255, 180/255),label='pFedKD', zorder=2)  # 使用RGB(31, 119, 180)
#plt.bar(xx + bar_width, y2, bar_width, color=(255/255, 127/255, 14/255),label='FedAVG', zorder=2)  # 使用RGB(255, 127, 14)
#plt.bar(x, diff, color='red', alpha=0.5, label='Difference')


for i in range(len(x)):
    plt.text(x[i], y1[i] + 0.3, f'{y1[i]:.2f}', ha='center', va='bottom',fontsize=9)
    plt.text(x[i] , y2[i] -2.5, f'{y2[i]:.2f}', ha='center', va='bottom',fontsize=9)

for i in range(len(x)):
    diff = y1[i] - y2[i]


plt.scatter(x, y1, color=(31/255, 119/255, 180/255),marker='s')
plt.scatter(x, y2, color=(255/255, 127/255, 14/255),marker='^')

# 设置 X 轴和 Y 轴的标签
bwith = 1.5  # 边框宽度设置为2
TK = plt.gca()  # 获取边框
TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
TK.spines['left'].set_linewidth(bwith)  # 图框左边
TK.spines['top'].set_linewidth(bwith)  # 图框上边
TK.spines['right'].set_linewidth(bwith)  # 图框右边
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.gcf()
plt.grid()
#plt.legend()
plt.xlabel('INPUT OF SIZE 2^12')
plt.ylabel('RUN TIME (MS)')
min_acc = 40
max_acc = 1000000
plt.ylim(min_acc, max_acc)
# 设置图表标题
plt.title('777777777777777777777777777')
#网格线
plt.grid(True, color='lightgray',linestyle='--', zorder=1)





fig_save_path = os.path.join('figs', '-'+ 'local training time'+ '.png')
plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0.2, format='png', dpi=400)
print('file saved to {}'.format(fig_save_path))

