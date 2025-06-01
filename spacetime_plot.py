import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# 顧客資料 
x = np.array([2.7 , 2 , 1.8 , 1 , 3 , 3.5 , 4 , 4.2 , 5.1 , 5])
y = np.array([2.8 , 3 , 2.5 , 2 , 1 , 1.5 , 5 , 4.8 , 4.3 , 4])
time = np.array([1,2,3,4,5,6,1,2,3,4])  # 假設的時間資料
arrival_time = np.array([1,5,3,4,2,6 , 1,2,3,4])  # 假設的到達時間
window_start = arrival_time - 1  # 每個人時間窗提前 1 單位開始
window_duration = 2.5
window_end = window_start + window_duration

# 模擬右圖（略為變動）
x_b = np.array([2.7 , 3 , 1.8 , 1 , 2 , 3.5 , 4 , 4.2 , 5.1 , 5])
y_b = np.array([2.8 , 1 , 2.5 , 2 , 3 , 1.5 , 5 , 4.8 , 4.3 , 4])
arrival_time_b = np.array([1,2,3,4,5,6 , 1,2,3,4])
window_start_b = arrival_time_b - 1
window_end_b = window_start_b + window_duration

# 倉庫設定
warehouse_x = (x.min() + x.max()) / 2 #3.05
warehouse_y = (y.min() + y.max()) / 2 #3.0
warehouse_z = 10
# print(warehouse_y)
def plot_cluster_area(ax, cx, cy):
    theta = np.linspace(0, 2 * np.pi, 100)
    radius_x, radius_y = 1.5, 2
    x_ellipse = cx + radius_x * np.cos(theta)
    y_ellipse = cy + radius_y * np.sin(theta)
    z_ellipse = np.zeros_like(theta)
    ax.plot(x_ellipse, y_ellipse, z_ellipse, 'r--', linewidth=2)

fig = plt.figure(figsize=(18, 10))

# 左圖 (a)
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("(a) Vehicle routing considering only geographical distance")
for i in range(len(x)):
    ax1.bar3d(x[i], y[i], window_start[i], 0.1, 0.1, window_duration, color='yellow', alpha=0.8, edgecolor='black')
    # 時間窗虛線
    ax1.plot([x[i], x[i]], [y[i], y[i]], [window_start[i], window_end[i]], 'r--', linewidth=1)
for i in [1,4]:
    ax1.bar3d(x[i], y[i], window_start[i], 0.1, 0.1, window_duration, color='red', alpha=1.0, edgecolor='black')
# 原始前6個點
x_part = x[:6]
y_part = y[:6]

# 在前後各加一個 0
x_with_zeros = np.insert(x_part, 0, warehouse_x)   # 前面加 0
x_with_zeros = np.append(x_with_zeros, warehouse_x)  # 後面加 0

y_with_zeros = np.insert(y_part, 0, warehouse_x)   # 前面加 0
y_with_zeros = np.append(y_with_zeros, warehouse_x)  # 後面加 0
ax1.plot(x_with_zeros, y_with_zeros, zs=0, zdir='z', color='black', linewidth=2)
time_part = time[:6]
time_part = np.insert(time_part, 0, 0)   # 前面加 0
time_part = np.append(time_part, 5)  # 後面加 0
ax1.plot(x_with_zeros, y_with_zeros, time_part, color='skyblue', linewidth=2)


x_part = x[6:10]
y_part = y[6:10]
# 在前後各加一個 0
x_with_zeros = np.insert(x_part, 0, warehouse_x)   # 前面加 0
x_with_zeros = np.append(x_with_zeros, warehouse_x)  # 後面加 0

y_with_zeros = np.insert(y_part, 0, warehouse_x)   # 前面加 0
y_with_zeros = np.append(y_with_zeros, warehouse_x)  # 後面加 0
ax1.plot(x_with_zeros, y_with_zeros, zs=0, zdir='z', color='black', linewidth=2)

time_part = time[6:10]
time_part = np.insert(time_part, 0, 0)   # 前面加 0
time_part = np.append(time_part, 7)  # 後面加 0

ax1.plot(x_with_zeros, y_with_zeros, time_part, color='skyblue', linewidth=2)
for xi, yi in zip(x, y):
    ax1.scatter(xi, yi, 0, marker='P', color='black', s=60)
ax1.bar3d(warehouse_x, warehouse_y, 0, 0.15, 0.15, 10, color='green', alpha=0.8, edgecolor='black')
ax1.scatter(warehouse_x, warehouse_y, 0, marker='X', color='green', s=100)
plot_cluster_area(ax1, 2.5, 2.5)
plot_cluster_area(ax1, 4.5, 4.5)
ax1.set_xlabel('Space X')
ax1.set_ylabel('Space Y')
ax1.set_zlabel('Time')
ax1.view_init(elev=30, azim=-45)
ax1.set_zlim(0, max(window_end) + 2)
ax1.set_xlim(x.min() - 1, x.max() + 1)
ax1.set_ylim(y.min() - 1, y.max() + 1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax1.grid(False)

# 右圖 (b)
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("(b) Vehicle routing considering spatiotemporal distance")
for i in range(len(x_b)):
    ax2.bar3d(x_b[i], y_b[i], window_start_b[i], 0.1, 0.1, window_duration, color='yellow', alpha=0.8, edgecolor='black')
# 原始前6個點
x_part = x_b[:6]
y_part = y_b[:6]

# 在前後各加一個 0
x_with_zeros = np.insert(x_part, 0, warehouse_x)   # 前面加 0
x_with_zeros = np.append(x_with_zeros, warehouse_x)  # 後面加 0

y_with_zeros = np.insert(y_part, 0, warehouse_x)   # 前面加 0
y_with_zeros = np.append(y_with_zeros, warehouse_x)  # 後面加 0
ax2.plot(x_with_zeros, y_with_zeros, zs=0, zdir='z', color='black', linewidth=2)
time_part = time[:6]
time_part = np.insert(time_part, 0, 0)   # 前面加 0
time_part = np.append(time_part, 7)  # 後面加 0
ax2.plot(x_with_zeros, y_with_zeros, time_part, color='skyblue', linewidth=2)
x_part = x_b[6:10]
y_part = y_b[6:10]
# 在前後各加一個 0
x_with_zeros = np.insert(x_part, 0, warehouse_x)   # 前面加 0
x_with_zeros = np.append(x_with_zeros, warehouse_x)  # 後面加 0

y_with_zeros = np.insert(y_part, 0, warehouse_x)   # 前面加 0
y_with_zeros = np.append(y_with_zeros, warehouse_x)  # 後面加 0
ax2.plot(x_with_zeros, y_with_zeros, zs=0, zdir='z', color='black', linewidth=2)
time_part = time[6:10]
time_part = np.insert(time_part, 0, 0)   # 前面加 0
time_part = np.append(time_part, 5)  # 後面加 0
ax2.plot(x_with_zeros, y_with_zeros, time_part, color='skyblue', linewidth=2)
for xi, yi in zip(x_b, y_b):
    ax2.scatter(xi, yi, 0, marker='P', color='black', s=60)
ax2.bar3d(warehouse_x, warehouse_y, 0, 0.15, 0.15, 10, color='green', alpha=0.8, edgecolor='black')
ax2.scatter(warehouse_x, warehouse_y, 0, marker='X', color='green', s=100)
plot_cluster_area(ax2, 2.0, 2.0)
plot_cluster_area(ax2, 4.5, 4.5)
ax2.set_xlabel('Space X')
ax2.set_ylabel('Space Y')
ax2.set_zlabel('Time')
ax2.view_init(elev=30, azim=-45)
ax2.set_zlim(0, max(window_end_b) + 2)
ax2.set_xlim(x_b.min() - 1, x_b.max() + 1)
ax2.set_ylim(y_b.min() - 1, y_b.max() + 1)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
ax2.grid(False)
# 圖例
legend_elements = [
    mpatches.Patch(color='yellow', label='Time windows'),
    mpatches.Patch(color='red', label='Violation of time windows'),
    mpatches.Patch(color='green', label='Satellite time window'),
    mpatches.Patch(facecolor='none', edgecolor='red', linestyle='--', label='Customer clustering'),
    mlines.Line2D([], [], color='black', linewidth=2, label='Actual path'),
    mlines.Line2D([], [], color='skyblue', linewidth=2, label='Space-time path'),
    mlines.Line2D([], [], marker='P', color='black', linestyle='None', markersize=10, label='Customer'),
    mlines.Line2D([], [], marker='X', color='green', linestyle='None', markersize=10, label='Satellite')
]

fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize='large', frameon=False, bbox_to_anchor=(0.5, -0.0001))

plt.subplots_adjust(top=1.1, bottom=0.005)

# 匯出圖檔
plt.savefig("clustering_time_windows_variable_start.png", dpi=300, bbox_inches='tight')
plt.show()