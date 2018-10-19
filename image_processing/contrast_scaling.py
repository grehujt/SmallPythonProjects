import numpy as np
import cv2

im = cv2.imread('testim.png')
im[im<50] = 50  # for illustration purpose
im[im>200] = 200  # for illustration purpose
rmax, rmin = np.max(im, axis=(0, 1)), np.min(im, axis=(0, 1))
# scale pixel range from [rmin, rmax] to [0, 255]
im2 = (((im-rmin)/(rmax-rmin))*255).astype('uint8')
cv2.imshow('im', im)
cv2.imshow('im2', im2)
cv2.waitKey(0)
