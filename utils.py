import cv2

def draw_chessboard_corners(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,5), None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        image = cv2.drawChessboardCorners(image, (7,5), corners2, ret)
    return image