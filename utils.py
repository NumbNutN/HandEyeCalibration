import cv2
import numpy as np

def DEBUG_PRINT(*args, **kwargs):
    with open("debug.log", "a") as f:
        print(*args, **kwargs, file=f)

        
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

def draw_chessboard_corners(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,5), None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        image = cv2.drawChessboardCorners(image, (7,5), corners2, ret)
    return image