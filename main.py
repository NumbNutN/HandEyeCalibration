import cv2
import numpy as np
import os
import h5py
import re

from utils import *


def generate_arithmetic_sequence(num_items,begin = 0,end = 199):
    if num_items <= 0:
        return []
    step = (end-begin) / (num_items - 1)
    sequence = [begin + round(i * step) for i in range(num_items)]
    
    return sequence


def read_data_from_hdf5(folder_path, camera_type:str, index_list:list):
    file_names = os.listdir(folder_path)
    
    hdf5_data = {
        'sim_qpos': [],
        'position': [],
        'rotation': [],
        'image': [],
    }

    HILIGHT_PRINT("Reading data from hdf5 files...")
    

    hdf5_files = [f for f in file_names if f.endswith('.hdf5')]
    hdf5_files_sorted = sorted(hdf5_files, key=lambda x: int(re.search(r'episode_(\d+).hdf5', x).group(1)))


    # hdf5_files_sorted= ["episode_2.hdf5"]
    HILIGHT_PRINT("read file:", hdf5_files_sorted)

    for file_name in hdf5_files_sorted:
        file_path = os.path.join(folder_path, file_name)
        with h5py.File(file_path, 'r') as file:

            
            print(list(file.keys()))  # 打印['action', 'base_action', 'observations']
            
            observations = file["observations"]
            
            qpos_list = observations['qpos']
            if camera_type == 'high':
                images_list = unzip(observations['images']['cam_high'])
            elif camera_type == 'left':
                images_list = unzip(observations['images']['cam_left_wrist'])
            else: # cam_right_wrist
                images_list = unzip(observations['images']['cam_right_wrist'])

            print(f"total qpos number in {file_name} :", len(images_list))

            if index_list is None:
                index_list = np.arange(len(images_list))
            
            print("sample number:", len(index_list))

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
                if sharpness < 400:
                    continue
                else:
                    hdf5_data['sim_qpos'].append(qpos)
                    hdf5_data['image'].append(image)

        index_list = None
    
    print(f"actual qpos number: {len(hdf5_data['image'])}")
    return hdf5_data # len == 20


def detect_image_corners(image_list, chessboard_size=(7,5),square_size=0.025):

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
            objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2) * square_size

            ## DEBUG
            # DEBUG_PRINT(f"Image {idx} has chessboard corners detected")
            # DEBUG_PRINT("objp", objp)
            # DEBUG_PRINT("corners2", corners2)

            #! WARNING the sequence of the corners will cause calibration error
            ## keep the same sequence as objp
            # calculate the cosine of the angle between the two vectors
            v1 = (objp[1] - objp[0])[:2]
            v2 = (corners2[1] - corners2[0])[0]
            # print("shape",v1.shape,v2.shape)
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            if(cos < 0):
                corners2 = np.flip(corners2, axis=0)
                

            objps.append(objp)
            imgps.append(corners2)
        else:
            # print(f"Image {idx} has no chessboard corners detected")
            skip_list.append(idx)

    return objps, imgps, skip_list

def calculate_reprojection_error(objps,imgps,rvecs,tvecs,mtx,dist,log=True):

    # calculate re-projection error
    HILIGHT_PRINT("Calculating re-projection error ...")
    mean_error = 0
    errors = []
    imgps_reproject = []
    for i in range(len(objps)):
        imgpoints2, _ = cv2.projectPoints(objps[i], rvecs[i], tvecs[i], mtx, dist)
        imgps_reproject.append(imgpoints2)
        error = cv2.norm(imgps[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        # print(f"image{i} with error {error}")
        mean_error += error
        errors.append(error)

    mean_error /= len(objps)
    print(f"Total re-projection error: {mean_error:.4f}")

    return imgps_reproject,errors


def calculate_intrinsic(objps,imgps,image_size,mtx=None,dist=None):
    '''
    
    '''
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objps, imgps, image_size, cameraMatrix=mtx, distCoeffs=dist)
    
    if ret:
        print("\nIntrinsic matrix:", mtx,sep="\n")
        print("\nDistortion coefficients:", dist,sep=" ")
    else:
        print("Camera calibration failed")

    return mtx, dist

def solvePnp(objps,imgps,mtx,dist):
    '''
    using pnp to calculate the target2camera
    '''
    # calculate the rotation and translation vectors
    Ts = []
    rvecs = []
    tvecs = []
    for i in range(len(objps)):
        ret, rvec, tvec = cv2.solvePnP(objps[i], imgps[i], mtx, dist)
        T = np.eye(4)
        R = cv2.Rodrigues(rvec)[0]
        T[0:3, 0:3] = R
        T[0:3, 3] = tvec.flatten()
        Ts.append(T)
        rvecs.append(rvec)
        tvecs.append(tvec)
        
    return Ts, rvecs, tvecs

import time
def get_poses(gripper2base,images,chessboard_size=(7,5),square_size=0.025,mtx=None,dist=None):

    HILIGHT_PRINT("Detecting Chessboard Corner ...")
    start = time.time()

    objps, imgps, skip_list = detect_image_corners(images,chessboard_size,square_size)
    print(f"Skip list: {skip_list}")

    if len(objps) == 0:
        print("No chessboard corners detected in the images")
        return None


    HILIGHT_PRINT("end Corner detection, time:", time.time()-start)

    objps = np.array(objps)
    imgps = np.array(imgps)

    # store privious intrinsic matrix and distortion coefficients
    mtx_prev = mtx
    dist_prev = dist

    HILIGHT_PRINT("Calibrating camera ...")
    start = time.time()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objps, imgps, images[0].shape[1::-1], cameraMatrix=mtx, distCoeffs=dist)
    HILIGHT_PRINT("end camera calibration, time:", time.time()-start)

    imgps_reproject,errors = calculate_reprojection_error(objps,imgps,rvecs,tvecs,mtx,dist)

    print("\nIntrinsic matrix:", mtx,sep="\n")
    print("\nDistortion coefficients:", dist,sep=" ")

    # construct homogenous transformation matrix
    # !WARN  what is Rodrigues
    tar2cam = []
    for rvec, tvec in zip(rvecs, tvecs):
        T = np.eye(4)
        R = cv2.Rodrigues(rvec)[0]
        T[0:3, 0:3] = R
        T[0:3, 3] = tvec.flatten()
        # print("Target tvec:", tvec.flatten())

        # !WARN : in our pinhole camera frame ,x-axis is front, y-axis is left, z-axis is up
        T_cam2ourcam = np.eye(4)
        T_cam2ourcam[0:3, 0:3] = np.array([[0, 0, 1], [-1,0,0], [0, -1, 0]]) 
        tar2cam.append(T_cam2ourcam @ T)

    # filter out the skipped images
    gripper2base = [gripper2base[i] for i in range(len(gripper2base)) if i not in skip_list]

    # return filter image list
    images = [images[i] for i in range(len(images)) if i not in skip_list]

    return gripper2base,tar2cam,mtx,dist,images,imgps,imgps_reproject,errors



def calibration(gripper2base, target2cam):

    '''
        This is for Eye in Hand Pattern
    '''

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


def calibration_eye2hand(g2bs,t2cs):

    '''
        This is for Eye to Hand Pattern
    '''
    b2gs = [g2b.T for g2b in g2bs]
    R_b2g = [T[0:3, 0:3] for T in b2gs]
    R_t2c = [T[0:3, 0:3] for T in t2cs]
    t_b2g = [T[0:3, 3] for T in b2gs]
    t_t2c = [T[0:3, 3] for T in t2cs]

    # hand-eye calibration
    R_c2b, t_c2b = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK)

    T_c2b = np.eye(4)
    T_c2b[:3,:3] = R_c2b
    T_c2b[:3,3] = t_c2b.flatten()

    return  T_c2b

import time
from matplotlib import pyplot as plt
import transforms3d.quaternions as quat
import transforms3d.euler as euler
import mplib

from visualizer import TrajectoryPlotter

## Config

from yaml import load, dump,Loader, Dumper
import logging


def main():

    logging.basicConfig(level=logging.INFO)

    folder_path = "./data/board"

    HILIGHT_PRINT("Read config...")
    config = load(open('config.yaml'), Loader=Loader)

    mode = config['pattern']
    square_size = config['square_size']
    chessboard_size = tuple(config['chessboard_size'])
    mtx_initial = np.array(config['intrinsic_matrix']).reshape(3,3)
    dist_initial = np.array(config['distortion_coefficients']).reshape(1,5)


    HILIGHT_PRINT("Forward Kinematics...")
    planner = mplib.Planner(
        urdf="aloha/arx5_description_isaac_colored.urdf",
        srdf="aloha/arx5_description_isaac_colored.srdf",
        move_group="fl_link6",
    )
    initial_qpos = planner.robot.get_qpos()

    for idx in range(0,1):

        # sharp image has been excluded
        # info_dict = read_data_from_hdf5(folder_path, 'left', generate_arithmetic_sequence(100,idx,idx+400))
        
        selects = np.linspace(0, 47, 48, endpoint=False, dtype=int)
        info_dict = read_data_from_hdf5(folder_path, 'left', None)

        qposes, images = info_dict['sim_qpos'], info_dict['image']

        g2bs = []
        for qpos in qposes:
            all_qpos = initial_qpos.copy()
            all_qpos[planner.move_group_joint_indices] = qpos[:-1] # without gripper
            planner.pinocchio_model.compute_forward_kinematics(all_qpos)
            ee_pose = planner.pinocchio_model.get_link_pose(planner.move_group_link_id)
            g2bs.append(ee_pose.to_transformation_matrix())
        
        #! WARN rotation definition

        ### optimize the instrinstic matrix ###

        # g2bs,t2cs,mtx_initial,dist_initial,images, imgps,imgps_reproject,errors  = get_poses(g2bs, images, chessboard_size, square_size,mtx=mtx_initial,dist=dist_initial)
        

        #### using fixed instrinsic matrix ####
        
        HILIGHT_PRINT("Detecting Chessboard Corner ...")
        HILIGHT_PRINT("Using chessboard size:", chessboard_size)
        HILIGHT_PRINT("Using square size:", square_size)

        objps, imgps, skip_list = detect_image_corners(images,chessboard_size,square_size)
        print(f"Skip list: {skip_list}")

        if len(objps) == 0:
            print("No chessboard corners detected in the images")
            return None

        objps = np.array(objps)
        imgps = np.array(imgps)

        # calculate instrinsic matrix
        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objps, imgps, images[0].shape[1::-1], cameraMatrix=mtx_initial, distCoeffs=dist_initial)
        # print("\nIntrinsic matrix:", mtx,sep="\n")
        # print("\nDistortion coefficients:", dist,sep=" ")

        # store privious intrinsic matrix and distortion coefficients
        mtx = mtx_initial
        dist = dist_initial
        t2cs,rvecs,tvecs = solvePnp(objps,imgps,mtx,dist)

        imgps_reproject,errors = calculate_reprojection_error(objps,imgps,rvecs,tvecs,mtx_initial,dist_initial)

        # construct homogenous transformation matrix        
        trans = np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])
        t2cs = [trans @ T for T in t2cs]

        # filter out the skipped images
        g2bs = [g2bs[i] for i in range(len(g2bs)) if i not in skip_list]
        images = [images[i] for i in range(len(images)) if i not in skip_list]

        plotter = TrajectoryPlotter()


        if mode == 'eye-in-hand':
            c2g = calibration(g2bs, t2cs)

            ###########################
            ### write ground truth  ###
            ###########################
            # c2g = np.eye(4)
            # euler2 = np.deg2rad([-1, 30, 4])
            # c2g[0:3, 0:3] = quat.quat2mat(
            #     euler.euler2quat(euler2[0],euler2[1],euler2[2], 'sxyz') # rotation
            # )
            # c2g[0:3, 3] = np.array([0.07 , -0.01 , 0.095])


            ###########################
            c2bs = [g2b @ c2g for g2b in g2bs]
            t2bs = [c2b @ t2c for t2c, c2b in zip(t2cs, c2bs)]
            res = c2g


            ## show calibration info
            HILIGHT_PRINT("\nExtrinsics",res)
            pos = res[0:3, 3]
            rot = quat.mat2quat(res[:3, :3])
            print(f'Position: {pos}')
            print(f'Rotation: {rot}')

            # to euler angle
            euler2 = euler.quat2euler(rot,'sxyz')
            # to degree
            euler2 = np.rad2deg(euler2)
            print(f'Euler: {euler2}')

            # calculate the variance of target position
            target_pos = np.array([t2b[0:3, 3] for t2b in t2bs])
            target_pos_var = np.var(target_pos, axis=0)
            print(f"Target Position Variance: {target_pos_var}")

            ### bounding box
            rot_true = np.array([0.9820343852043152, 0.0042511820793151855, 0.18733306229114532, 0.022285446524620056])
            euler2 = np.rad2deg(euler.quat2euler(rot_true,'sxyz'))
            pos_true = np.array([0.0382, -0.018, 0.10315])
            print("bounding box:",euler2)

            ######################################
            ###    Visualize Trajectory        ###
            ######################################

            for i in range(len(c2bs)):
                
                # count time in the loop
                start = time.time()
                plotter.update_trajectory(1, g2bs[i], label='Gripper')            
                plotter.update_trajectory(2, c2bs[i], label='Camera')  
                plotter.update_trajectory(3, t2bs[i], label='Target')
                plotter.draw_image_and_chessboard_corners(1, images[i], imgps[i], pattern_size=chessboard_size)
                plotter.draw_image_and_chessboard_corners(2, images[i], imgps_reproject[i], pattern_size=chessboard_size)
                # print(f"Image {i} time: {time.time()-start:.4f}s")

                HILIGHT_PRINT(f"Image {i} re-projection error: {errors[i]:.4f}")
                HILIGHT_PRINT(f"Target Position: {t2bs[i][0:3, 3]}")

                start = time.time()
                plotter.update()
                # time.sleep(3)
                # print(f"Update time: {time.time()-start:.4f}s")

            ################################################
            ##     Visualize True and Estimated c2g       ##
            ################################################



            T_true = np.eye(4)
            T_true[:3,:3] = quat.quat2mat(rot_true)
            T_true[:3, 3] = pos_true

            plotter.draw_coordinate_axes(0,T_true, scale=0.1,label='Ground True')
            plotter.draw_coordinate_axes(1,c2g, scale=0.1,label='Estimated')





        if mode == 'eye-to-hand':
            c2b = calibration_eye2hand(g2bs,t2cs)

            t2gs = [b2g.T @ c2b @ t2c for b2g,t2c in zip(g2bs,t2cs)]
            res = c2b

            ## show calibration info
            HILIGHT_PRINT("\nExtrinsics",res)
            pos = res[0:3, 3]
            rot = quat.mat2quat(res[:3, :3])
            print(f'Position: {pos}')
            print(f'Rotation: {rot}')

            ######################################
            ###    Visualize Trajectory        ###
            ######################################

            for i in range(len(c2bs)):
                
                # count time in the loop
                start = time.time()
                plotter.update_trajectory(1, t2gs[i], label='Target2Gripper')
                plotter.draw_image_and_chessboard_corners(1, images[i], imgps[i])
                plotter.draw_image_and_chessboard_corners(2, images[i], imgps_reproject[i])
                # print(f"Image {i} time: {time.time()-start:.4f}s")

                # HILIGHT_PRINT(f"Image {i} re-projection error: {errors[i]:.4f}")

                start = time.time()
                plotter.update()
                # print(f"Update time: {time.time()-start:.4f}s")
        
    

            plt.show()

        ## Dump to yaml
        # config['intrinsic_matrix'] = mtx_initial.tolist()
        # config['distortion_coefficients'] = dist_initial.tolist()
        # config['extrinsics'] = c2g.tolist()
        # with open('config_gen.yaml', 'w') as f:
        #     dump(config, f, Dumper=Dumper)



if __name__ == '__main__':
    main()