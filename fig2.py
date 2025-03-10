# （FedGen-main）2023/09/16 17:36
# （FedGen-main）2023/09/14 22:39
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams.update({'font.size': 14})
#matplotlib.rcParams['font.family'] = 'Times New Roman'
# 定义 Y 轴的数据


#plt.figure(figsize=(6, 6))
plt.figure(1, figsize=(5, 5))


# 模拟数据
xx = ['BG=8', 'BG=16', 'BG=32', 'BG=64', 'BG=128']
yy1 = [75.70, 76.55, 78.03, 79.25, 79.05]
yy2 = [71.14, 71.14,71.14,71.14,71.14]

# 绘制柱状图
x = np.arange(len(xx))
bar_width = 0.35
plt.plot(xx, yy1,  linestyle='-', color=(31/255, 119/255, 180/255), zorder=3)
#plt.plot(xx+ bar_width, yy2, linestyle='-', color=(255/255, 127/255, 14/255), zorder=3)
plt.bar(x, yy1, bar_width, label='FedPKD', color=(31/255, 119/255, 180/255), zorder=2)  # 使用RGB(31, 119, 180)
plt.bar(x + bar_width, yy2, bar_width, label='FedAVG', color=(255/255, 127/255, 14/255), zorder=2)  # 使用RGB(255, 127, 14)
#plt.vlines(x + 0.175, np.minimum(yy1, yy2), np.maximum(yy1, yy2), colors='red', linestyles='dashed')
# 设置刻度标签和标题
plt.xlabel('x')
plt.ylabel('y')
plt.title('1111')
plt.xticks(x + bar_width / 2, xx)

# 添加标签
for i in range(len(xx)):
    plt.text(x[i], yy1[i] + 0.2, f'{yy1[i]:.2f}', ha='center', va='bottom',fontsize=9, zorder=3)
    plt.text(x[i] + bar_width + 0.04, yy2[i] +0.2, f'{yy2[i]:.2f}', ha='center', va='bottom',fontsize=9, zorder=3)

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
plt.xlabel('X')
plt.ylabel('Y')
min_acc = 68
max_acc = 81
plt.ylim(min_acc, max_acc)
# 设置图表标题
plt.title('88888')
#网格线
plt.grid(True, color='lightgray',linestyle='--', zorder=1)
# 显示图表

fig_save_path = os.path.join('figs', '-'+ '222222222222222222222'+ '.png')
plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0.2, format='png', dpi=400)
print('file saved to {}'.format(fig_save_path))
