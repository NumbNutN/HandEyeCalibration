import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryPlotter:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.trajectories = {}  # 存储多个质点的轨迹
        self.traj_labels = {}

        self.poses = {}  # 存储多个物体的位姿
        self.pose_labels = {}
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # 预定义颜色
        self.axis_colors = ['r', 'g', 'b', 'm', 'y', 'k']  # X, Y, Z 轴颜色

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Motion Trajectories')

    def update_trajectory(self, point_id, T, label=None):
        """
        更新质点的轨迹
        :param point_id: 质点的唯一标识
        :param T: 4x4 齐次变换矩阵
        """
        if point_id not in self.trajectories:
            self.trajectories[point_id] = []
            self.traj_labels[point_id] = label

        # 提取平移部分（X, Y, Z）
        position = T[:3, 3]
        self.trajectories[point_id].append(position)

        # 清除并重绘
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Motion Trajectories')

        # 绘制所有质点轨迹
        for i, (pid, traj) in enumerate(self.trajectories.items()):
            traj = np.array(traj)
            color = self.colors[i % len(self.colors)]  # 轮询颜色
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, label=self.traj_labels
                                                                                [pid] if self.traj_labels[pid] is not None else f'Point {pid}')
            self.ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=color, marker='o')

        self.ax.legend()
        plt.pause(0.1)  # 短暂停止以更新图像


    def draw_coordinate_axes(self, point_id, T, label=None,scale=1.0):
            """
            绘制物体的局部坐标系
            :param origin: 坐标系原点 (3,)
            :param R: 旋转矩阵 3x3
            :param scale: 坐标轴缩放因子
            """

            if point_id not in self.poses:
                self.poses[point_id] = []
                self.pose_labels[point_id] = label

            R = T[:3, :3]
            origin = T[:3, 3]

            axis_vectors = np.eye(3)  # 单位坐标系
            transformed_axes = R @ (axis_vectors * scale)  # 变换后的坐标系

            for i in range(3):
                start = origin
                end = origin + transformed_axes[:, i]
                self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                            color=self.axis_colors[i+point_id*3], linewidth=2, label=self.pose_labels[point_id] if ((self.pose_labels[point_id] is not None) and i == 0) else f'')
                
            self.ax.legend()

# 示例使用
if __name__ == "__main__":
    import time

    plotter = TrajectoryPlotter()

    # 生成随机的齐次变换矩阵，模拟运动
    for t in range(100):
        T1 = np.eye(4)
        T1[:3, 3] = [0.1 * t, np.sin(t * 0.1), np.cos(t * 0.1)]  # 质点 1 运动轨迹

        T2 = np.eye(4)
        T2[:3, 3] = [-0.1 * t, np.cos(t * 0.1), np.sin(t * 0.1)]  # 质点 2 运动轨迹

        plotter.update_trajectory(1, T1)
        plotter.update_trajectory(2, T2)

        time.sleep(0.1)  # 模拟实时更新

    plt.show()
