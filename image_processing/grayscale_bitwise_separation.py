import numpy as np
import cv2

im = cv2.imread('testim.png')
if len(im.shape) == 3:
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow('im', im)
cv2.imshow('im0', (np.bitwise_and(im, 0x01 << 0)*255).astype('uint8'))
cv2.imshow('im1', (np.bitwise_and(im, 0x01 << 1)*255).astype('uint8'))
cv2.imshow('im2', (np.bitwise_and(im, 0x01 << 2)*255).astype('uint8'))
cv2.imshow('im3', (np.bitwise_and(im, 0x01 << 3)*255).astype('uint8'))
cv2.imshow('im4', (np.bitwise_and(im, 0x01 << 4)*255).astype('uint8'))
cv2.imshow('im5', (np.bitwise_and(im, 0x01 << 5)*255).astype('uint8'))
cv2.imshow('im6', (np.bitwise_and(im, 0x01 << 6)*255).astype('uint8'))
cv2.imshow('im7', (np.bitwise_and(im, 0x01 << 7)*255).astype('uint8'))
cv2.waitKey(0)
