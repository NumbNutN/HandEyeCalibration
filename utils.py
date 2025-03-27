import cv2
import numpy as np

def DEBUG_PRINT(*args, **kwargs):
    with open("debug.log", "a") as f:
        print(*args, **kwargs, file=f)

def HILIGHT_PRINT(*args, **kwargs):
    print("\033[1;33m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")

        
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

def draw_chessboard_corners(image,chessboard_size = (7,5), square_size = 0.025):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,5), None)

    if ret:
        
        objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2) * square_size
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # to keep object points seq
        v1 = (objp[1] - objp[0])[:2]
        v2 = (corners2[1] - corners2[0])[0]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if(cos < 0):
            corners2 = np.flip(corners2, axis=0)

        image = cv2.drawChessboardCorners(image, (7,5), corners2, ret)
    return image