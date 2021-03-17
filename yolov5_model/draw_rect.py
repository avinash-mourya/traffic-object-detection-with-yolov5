# Python program to extract rectangular
# Shape using OpenCV in Python3
import cv2
import numpy as np

drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle.
ix, iy = -1, -1


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 3)
                a = x
                b = y
                if a != x | b != y:
                    cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 0), -1)
            else:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

        else:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
video_capture = cv2.VideoCapture("traffic.mp4")
ret, frame = video_capture.read()  # frame shape 640*360*3
while True:
    if ret != True:
        break
    cv2.imshow("image",img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()
