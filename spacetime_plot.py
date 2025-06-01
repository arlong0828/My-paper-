import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# 顧客資料（共 10 筆）
x = np.array([1, 2, 3, 4, 5, 1.8, 2.7, 3.5, 4.2, 5.1])
y = np.array([2, 3, 1, 5, 4, 2.5, 2.8, 1.5, 4.8, 4.3])
z = np.array([2, 4, 6, 3, 5, 3.2, 4.1, 5.8, 3.5, 6.2])

x_b = np.array([1, 2.2, 2.5, 3.8, 5, 1.7, 2.5, 3.3, 4.0, 5.2])
y_b = np.array([2, 2.5, 1.2, 4.5, 4, 2.2, 2.7, 1.6, 4.6, 4.1])
z_b = np.array([2, 3, 5.5, 4, 5.8, 3.0, 3.9, 5.5, 4.2, 6.0])

# 計算倉庫位置與時間窗
warehouse_x = (x.min() + x.max()) / 2
warehouse_y = (y.min() + y.max()) / 2
warehouse_z = 10

# 群聚範圍繪製函數
def plot_cluster_area(ax, cx, cy):
    theta = np.linspace(0, 2 * np.pi, 100)
    radius_x, radius_y = 1.5, 2
    x_ellipse = cx + radius_x * np.cos(theta)
    y_ellipse = cy + radius_y * np.sin(theta)
    z_ellipse = np.zeros_like(theta)
    ax.plot(x_ellipse, y_ellipse, z_ellipse, 'r--', linewidth=2)

# 建立圖形
fig = plt.figure(figsize=(18, 10))

# 左圖
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("(a) Clustering and vehicle routes\nconsidering geographical distances")
for i in range(len(x)):
    ax1.bar3d(x[i], y[i], 0, 0.1, 0.1, z[i], color='yellow', alpha=0.8, edgecolor='black')
for i in [2, 3, 7]:
    ax1.bar3d(x[i], y[i], 0, 0.1, 0.1, z[i], color='red', alpha=1.0, edgecolor='black')
ax1.plot(x, y, zs=0, zdir='z', color='black', linewidth=2)
ax1.plot(x, y, z, color='skyblue', linewidth=2)
for xi, yi in zip(x, y):
    ax1.scatter(xi, yi, 0, marker='P', color='black', s=60)
ax1.bar3d(warehouse_x, warehouse_y, 0, 0.15, 0.15, warehouse_z, color='green', alpha=0.8, edgecolor='black')
ax1.scatter(warehouse_x, warehouse_y, 0, marker='X', color='green', s=100)
plot_cluster_area(ax1, 2.5, 2.5)
plot_cluster_area(ax1, 4.5, 4.5)
ax1.set_xlabel('Space X')
ax1.set_ylabel('Space Y')
ax1.set_zlabel('Time')
ax1.view_init(elev=30, azim=-45)

# 右圖
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("(b) Clustering and vehicle routes\nconsidering space-time distances")
for i in range(len(x_b)):
    ax2.bar3d(x_b[i], y_b[i], 0, 0.1, 0.1, z_b[i], color='yellow', alpha=0.8, edgecolor='black')
ax2.plot(x_b, y_b, zs=0, zdir='z', color='black', linewidth=2)
ax2.plot(x_b, y_b, z_b, color='skyblue', linewidth=2)
for xi, yi in zip(x_b, y_b):
    ax2.scatter(xi, yi, 0, marker='P', color='black', s=60)
ax2.bar3d(warehouse_x, warehouse_y, 0, 0.15, 0.15, warehouse_z, color='green', alpha=0.8, edgecolor='black')
ax2.scatter(warehouse_x, warehouse_y, 0, marker='X', color='green', s=100)
plot_cluster_area(ax2, 2.0, 2.0)
plot_cluster_area(ax2, 4.5, 4.5)
ax2.set_xlabel('Space X')
ax2.set_ylabel('Space Y')
ax2.set_zlabel('Time')
ax2.view_init(elev=30, azim=-45)

# 圖例
legend_elements = [
    mpatches.Patch(color='yellow', label='Time windows'),
    mpatches.Patch(color='red', label='Violation of time windows'),
    mpatches.Patch(color='green', label='Warehouse time window'),
    mpatches.Patch(facecolor='none', edgecolor='red', linestyle='--', label='Customer clustering'),
    mlines.Line2D([], [], color='black', linewidth=2, label='Actual path'),
    mlines.Line2D([], [], color='skyblue', linewidth=2, label='Space-time path'),
    mlines.Line2D([], [], marker='P', color='black', linestyle='None', markersize=10, label='Customer'),
    mlines.Line2D([], [], marker='X', color='green', linestyle='None', markersize=10, label='Warehouse')
]

fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize='large', frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# 匯出 PNG
plt.savefig("clustering_with_warehouse.png", dpi=300, bbox_inches='tight')
plt.show()
