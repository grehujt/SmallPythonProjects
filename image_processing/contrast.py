import cv2
im = cv2.imread('testim.png')
im2 = (((im2/255.0)**0.37)*255).astype('uint8')
cv2.imshow('im', im)
cv2.imshow('im2', im2)
cv2.waitKey(0)
