import argparse
import cv2
import numpy as np
import os
import h5py
import re
import sys

import modern_robotics as mr

def unzip(images):
    images_list = []
    for bytes in images:
        img_array = np.frombuffer(bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        images_list.append(img)
    return np.array(images_list)


def check_image_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance_of_laplacian = laplacian.var()
    
    return variance_of_laplacian

def generate_arithmetic_sequence(num_items):
    if num_items <= 0:
        return []
    step = 199 / (num_items - 1)
    sequence = [round(i * step) for i in range(num_items)]
    
    return sequence

def compute_relative_transforms(poses):
    """
    计算连续两帧之间的相对运动
    对于一系列绝对齐次变换矩阵，返回每一对相邻帧的相对变换 T_rel = inv(T_i) * T_{i+1}
    """
    relative_transforms = []
    for i in range(len(poses) - 1):
        T_inv = np.linalg.inv(poses[i])
        # T_rel = np.dot(T_inv, poses[i+1])
        T_rel = T_inv @ poses[i+1] 
        relative_transforms.append(T_rel)
    return relative_transforms

def read_data_from_hdf5(folder_path, camera_type:str, index_list:list):
    file_names = os.listdir(folder_path)
    
    hdf5_data = {
        'sim_qpos': [],
        'position': [],
        'rotation': [],
        'image': [],
    }
    hdf5_files = [f for f in file_names if f.endswith('.hdf5')]
    hdf5_files_sorted = sorted(hdf5_files, key=lambda x: int(re.search(r'episode_(\d+).hdf5', x).group(1)))
    for file_name in hdf5_files_sorted:
        file_path = os.path.join(folder_path, file_name)
        with h5py.File(file_path, 'r') as file:
            # print(list(file.keys()))  # 打印['action', 'base_action', 'observations']
            
            observations = file["observations"]
            
            qpos_list = observations['qpos']
            if camera_type == 'high':
                images_list = unzip(observations['images']['cam_high'])
            elif camera_type == 'left':
                images_list = unzip(observations['images']['cam_left_wrist'])
            else: # cam_right_wrist
                images_list = unzip(observations['images']['cam_right_wrist'])
                
            for index in index_list:
                qpos = qpos_list[index, :]
                    # Extract the gripper qpos values from the 7th column

                #! WARN qpos definition
                if camera_type == 'left':
                    qpos = qpos[:7]
                else:
                    qpos = qpos[7:]

                image = images_list[index, :, :, :] # (480, 640, 3)
                
                # 检查image清晰度
                sharpness = check_image_sharpness(image)
                # print(sharpness)
                if sharpness < 600:
                    continue
                else:
                    hdf5_data['sim_qpos'].append(qpos)
                    hdf5_data['image'].append(image)
    # print(f"Expected length of images_list is {len(images_list)*4}")
    print(f"The length of images_list is {len(hdf5_data['image'])}")
    return hdf5_data # len == 20


def detect_image_corners(image_list, chessboard_size=(7,5)):

    # !WARN: chessboard_size 
    objps = []
    imgps = []

    skip_list = []

    for idx, image in enumerate(image_list):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
            objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)
            objps.append(objp)
            imgps.append(corners2)
        else:
            print(f"Image {idx} has no chessboard corners detected")
            skip_list.append(idx)

    return objps, imgps, skip_list


def get_poses(gripper_poses,image_list,square_size=0.025):

    objps, imgps, skip_list = detect_image_corners(image_list)
    print(f"Skip list: {skip_list}")

    if len(objps) == 0:
        print("No chessboard corners detected in the images")
        return None

    objps = np.array(objps)
    imgps = np.array(imgps)

    # intrinsic

    # !WARN what is imageSize
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objps, imgps, image_list[0].shape[1::-1], None, None)

    # construct homogenous transformation matrix
    # !WARN  what is Rodrigues
    target_poses = []
    for rvec, tvec in zip(rvecs, tvecs):
        T = np.eye(4)
        R = cv2.Rodrigues(rvec)[0]
        T[0:3, 0:3] = R
        T[0:3, 3] = tvec.flatten()
        # print("Target tvec:", tvec.flatten())
        target_poses.append(T)

    # filter out the skipped images
    gripper_poses = [gripper_poses[i] for i in range(len(gripper_poses)) if i not in skip_list]

    return gripper_poses,target_poses


def hand_eye_calibration(gripper_poses, target_poses):
    # compute relative transforms
    gripper2base = compute_relative_transforms(gripper_poses)
    target2cam = compute_relative_transforms(target_poses)

    R_gripper2base = [T[0:3, 0:3] for T in gripper2base]
    R_target2cam = [T[0:3, 0:3] for T in target2cam]
    t_gripper2base = [T[0:3, 3] for T in gripper2base]
    t_target2cam = [T[0:3, 3] for T in target2cam]

    # hand-eye calibration
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_PARK)

    # construct extrinsic matrix
    T_cam2gripper = np.eye(4)
    T_cam2gripper[0:3, 0:3] = R_cam2gripper
    T_cam2gripper[0:3, 3] = t_cam2gripper.flatten()

    return T_cam2gripper


#! LEFT or RIGHT
class vx300s:
    Slist = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
            [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
            [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
            [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
        ]
    ).T

    M = np.array(
        [
            [1.0, 0.0, 0.0, 0.536494],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.42705],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

from visualizer import TrajectoryPlotter
import time
from matplotlib import pyplot as plt

def main():

    folder_path = "./calibration"

    # sharp image has been excluded
    info_dict = read_data_from_hdf5(folder_path, 'left', generate_arithmetic_sequence(50))

    qpos_list, image_list = info_dict['sim_qpos'], info_dict['image']

    ee_poses = []

    #! WARN do qpos need a bias?

    for i in range(len(qpos_list)):
        
        T = mr.FKinSpace(vx300s.M, vx300s.Slist, qpos_list[i][:6])
        ee_poses.append(T)

        #! WARN rotation definition

    gripper_poses,target_poses  = get_poses(ee_poses, image_list)


    ### Visualizer
    plotter = TrajectoryPlotter()
    
    for i in range(len(gripper_poses)):

        plotter.update_trajectory(1, gripper_poses[i], label='Gripper')
        plotter.update_trajectory(2, target_poses[i], label='Target')

        time.sleep(0.1)  # 模拟实时更新

    plt.show()

    ext = hand_eye_calibration(gripper_poses, target_poses)

    print(ext)


if __name__ == '__main__':
    main()