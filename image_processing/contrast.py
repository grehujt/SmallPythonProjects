import cv2

im = cv2.imread('testim.png')
# 0.37 < 1, emphasis on the dark part
im2 = (((im/255.0)**0.37)*255).astype('uint8')
cv2.imshow('im', im)
cv2.imshow('im2', im2)
cv2.waitKey(0)
