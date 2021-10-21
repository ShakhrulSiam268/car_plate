import cv2

img=cv2.imread('out.png')
cv2.imshow('Frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()