import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryPlotter:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(221, projection='3d')
        

        self.ax_img1 = self.fig.add_subplot(223)
        self.ax_img1.set_title("Detect Corners")
        self.ax_img2 = self.fig.add_subplot(224)
        self.ax_img2.set_title("reProject Corners")
        

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

        self.ax_pose = self.fig.add_subplot(222, projection='3d')
        self.ax_pose.set_xlabel('X')
        self.ax_pose.set_ylabel('Y')
        self.ax_pose.set_zlabel('Z')
        self.ax_pose.set_title('3D Coordinate Axes')

        # 只更新变化的部分
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)


    def update_trajectory(self, point_id, T, label=None,scale=0.05):
        """
        更新质点的轨迹
        :param point_id: 质点的唯一标识
        :param T: 4x4 齐次变换矩阵
        """
        if point_id not in self.trajectories:
            self.trajectories[point_id] = {
                'T': [],
                'label': label,
                'line3d': self.ax.plot([], [], [], color=self.colors[point_id % len(self.colors)], label=label),
                'scatter3d': self.ax.scatter([], [], [], color=self.colors[point_id % len(self.colors)], marker='o')
            }

        # 提取平移部分（X, Y, Z）
        self.trajectories[point_id]['T'].append(T)

        # 清除并重绘
        # self.ax.clear()
        # self.ax.set_xlabel('X')
        # self.ax.set_ylabel('Y')
        # self.ax.set_zlabel('Z')
        # self.ax.set_title('3D Motion Trajectories')

        # 只绘制更新部分
        pos_array = np.array(self.trajectories[point_id]['T'])[:, :3, 3]  # 提取平移部分
        self.trajectories[point_id]['line3d'][0].set_data(pos_array[:, 0], pos_array[:, 1])
        self.trajectories[point_id]['line3d'][0].set_3d_properties(pos_array[:, 2])
        self.trajectories[point_id]['scatter3d'].remove()
        self.trajectories[point_id]['scatter3d'] = self.ax.scatter(pos_array[-1, 0], pos_array[-1, 1], pos_array[-1, 2],
                                                                    color=self.colors[point_id % len(self.colors)], marker='o')
        self.ax.legend()

        # 绘制所有质点轨迹
        # for i, (pid, traj) in enumerate(self.trajectories.items()):
        #     trans = np.array(traj)[:, :3, 3]  # 提取平移部分
        #     color = self.colors[i % len(self.colors)]  # 轮询颜色
        #     self.ax.plot(trans[:, 0], trans[:, 1], trans[:, 2], color=color, label=self.traj_labels
        #                                                                         [pid] if self.traj_labels[pid] is not None else f'Point {pid}')
        #     self.ax.scatter(trans[-1, 0], trans[-1, 1], trans[-1, 2], color=color, marker='o')

            # print(f' idx {i} Point {pid} Position: {traj}')

            # for j in range(3):
            #     start = traj[-1][:3, 3]
            #     end = start + scale * traj[-1][:3, j]
            #     self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
            #                 color=self.axis_colors[j], linewidth=2)

    def update(self):
        # self.fig.canvas.restore_region(self.background)  # 恢复背景

        # for point_id, traj in self.trajectories.items():
        #     # 绘制轨迹
        #     self.ax.draw_artist(traj['line3d'][0])
        #     self.ax.draw_artist(traj['scatter3d'])
        
        # self.fig.canvas.blit(self.fig.bbox)  # 更新绘图区域
        plt.pause(0.001)  # 暂停以更新图形


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
            self.ax_pose.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                        color=self.axis_colors[i+point_id*3], linewidth=2, label=self.pose_labels[point_id] if ((self.pose_labels[point_id] is not None) and i == 0) else f'')
            
        self.ax_pose.legend()


    def draw_image_and_chessboard_corners(self, id,image, corners):
        """
        """
        import cv2
        image = cv2.drawChessboardCorners(image, (7, 5), corners, True)

        # resize image to lower resolution for better visualization
        image = cv2.resize(image, (128, 96))
        if(id == 1):
            self.ax_img1.imshow(image)
        elif(id == 2):
            self.ax_img2.imshow(image)

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
