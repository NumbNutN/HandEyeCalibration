## intrinsic


#### Method

##### 自行标定

1. 不同角度和距离的图像

+ 不同角度和距离拍摄15-20张图像，确保覆盖摄像头视野多个距离

+ 从trajectory数据采样获取图像

2. 由`cv::calibrateCamera`优化内参

##### or 找已有的内参数据


## extrinsic

#### Method

1. 由正运动学从qpos推导每张图片机械臂Gripper的位姿序列$\{^b\mathrm{T}_g{}^{(1)},^b\mathrm{T}_g{}^{(2)},...\}$
2. 由`cv::calibrateCamera`获取照片中标定板相对于Camera的位姿$\{{}^c\mathrm{T}_t{}^{(1)},{}^c\mathrm{T}_t{}^{(2)},...\}$
> [opencv - calibrateCamera](https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d)


3. 由`cv::calibrateHandEye`求解${}^g\mathrm{T}_c{}$

$$
\begin{gathered} ^b\mathrm{T}_g{}^{(1)} {}^g\mathrm{T}_c{}^c\mathrm{T}_t{}^{(1)}={}^b\mathrm{T}_g{}^{(2)}{}^g\mathrm{T}_c{}^c\mathrm{T}_t{}^{(2)} \\ ({}^b\mathrm{T}_g{}^{(2)})^{-1b}\mathrm{T}_g{}^{(1)g}\mathrm{T}_c={}^g\mathrm{T}_c{}^c\mathrm{T}_t{}^{(2)}({}^c\mathrm{T}_t{}^{(1)})^{-1} \\ \mathrm{A}_{i}\mathrm{X}=\mathrm{XB}_i \end{gathered}
$$

> [opencv - calibrateHandEye](https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b)


播放HDF5 an Example
```
python3 visualize_episodes.py --dataset_dir "data" --save_dir "visualize" --episode_idx "2" --task_name ""
```