#!/usr/bin/python2
# -*- coding: utf-8 -*-
# @Time    : 2020/11/5 9:46
# @Author  : Caibojun
# @Email   : caibojun@myhexin.com
# @File    : area_polygen.py
# @Software: PyCharm

import os
import glob
import cv2
import numpy as np


def remove_small_objs(img):
    nb_components ,output ,stats ,centroids=cv2.connectedComponentsWithStats(img ,connectivity=4)
    sizes=stats[1: ,-1]
    nb_components=nb_components - 1
    min_size=int(1e-3 * output.shape[0] * output.shape[1])
    # your answer image
    result=np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0 ,nb_components):
        if sizes[i] >= min_size:
            result[output == i + 1]=255
    return result


def fillHole(im_in):
    im_floodfill=im_in.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h ,w=im_in.shape[:2]
    mask=np.zeros((h + 2 ,w + 2) ,np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill ,mask ,(0 ,0) ,255);

    # Invert floodfilled image
    im_floodfill_inv=cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out=im_in | im_floodfill_inv

    return im_out


def change_lr(x):
    rgb.l_r=x
    result=paraments_change(rgb)
    cv2.imshow("img" ,result)


def change_lg(x):
    rgb.l_g=x
    result=paraments_change(rgb)
    cv2.imshow("img" ,result)


def change_lb(x):
    rgb.l_b=x
    result=paraments_change(rgb)
    cv2.imshow("img" ,result)


def change_hr(x):
    rgb.h_r=x
    result=paraments_change(rgb)
    cv2.imshow("img" ,result)


def change_hg(x):
    rgb.h_g=x
    result=paraments_change(rgb)
    cv2.imshow("img" ,result)


def change_hb(x):
    rgb.h_b=x
    result=paraments_change(rgb)
    cv2.imshow("img" ,result)


def paraments_change(rgb):
    low=np.array([rgb.l_b ,rgb.l_g ,rgb.l_r])
    high=np.array([rgb.h_b ,rgb.h_g ,rgb.h_r])
    curr_mask=cv2.inRange(img_bgr ,low ,high)
    img_hsv[curr_mask == 0]=0
    img_hsv[curr_mask > 0]=([255 ,255 ,255])
    # viewImage(img_hsv)
    img_rgb=cv2.cvtColor(img_hsv ,cv2.COLOR_HSV2RGB)
    gray=cv2.cvtColor(img_rgb ,cv2.COLOR_RGB2GRAY)
    img_close=cv2.morphologyEx(gray ,cv2.MORPH_CLOSE ,cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3 ,3)))
    img_nohole=fillHole(img_close)
    img_nopiece=remove_small_objs(img_nohole)
    return img_nopiece

class Range:
    l_r = 0
    l_g = 0
    l_b = 0
    h_r = 255
    h_g = 255
    h_b = 255
rgb = Range()
font = cv2.FONT_HERSHEY_SIMPLEX
if __name__=='__main__':

    path = './esd_thumb/*.png'
    files = glob.glob(path)



    for i in files:
        rgb = Range
        color_range = None
        img_bgr = cv2.imread(i)
        cv2.imshow('raw',img_bgr)
        img_gauss = cv2.GaussianBlur(img_bgr,(7,7),0)
        try:
            img_hsv = cv2.cvtColor(img_gauss,cv2.COLOR_BGR2HSV)
        except Exception as e:
            print(i,e)
            continue
        name = os.path.basename(i)
        cv2.namedWindow(name,0)

        cv2.resizeWindow(name ,700 ,300)
        color_range_path = i.replace('.png','.txt')
        if os.path.isfile(color_range_path):
            color_range = np.loadtxt(color_range_path)
            rgb.l_b = int(color_range[0,0])
            rgb.l_g = int(color_range[0,1])
            rgb.l_r = int(color_range[0,2])
            rgb.h_b = int(color_range[1,0])
            rgb.h_g = int(color_range[1,1])
            rgb.h_r = int(color_range[1,2])
            default_img = paraments_change(rgb)
            cv2.imshow('img',default_img)


        # 第一个参数是滑动杆名称，第二个是对应的图片，第三个是默认值，第四个是最大值，第五个是回调函数
        cv2.createTrackbar('low red' ,name ,0 ,255 ,change_lr)
        cv2.createTrackbar('low green' ,name ,0 ,255 ,change_lg)
        cv2.createTrackbar('low blue' ,name ,0 ,255 ,change_lb)

        cv2.createTrackbar('high red' ,name ,0 ,255 ,change_hr)
        cv2.createTrackbar('high green' ,name ,0 ,255 ,change_hg)
        cv2.createTrackbar('high blue' ,name ,0 ,255 ,change_hb)

        cv2.setTrackbarPos('low red',name,rgb.l_r)
        cv2.setTrackbarPos('low green',name,rgb.l_g)
        cv2.setTrackbarPos('low blue',name,rgb.l_b)
        print(rgb.l_g)

        cv2.setTrackbarPos('high red',name,rgb.h_r)
        cv2.setTrackbarPos('high green',name,rgb.h_g)
        cv2.setTrackbarPos('high blue',name,rgb.h_b)
        while (1):
            # 拿到对应滑动杆的值
            if color_range is None:
                l_r=cv2.getTrackbarPos('low red' ,name)
                l_g=cv2.getTrackbarPos('low green' ,name)
                l_b=cv2.getTrackbarPos('low blue' ,name)
                h_r=cv2.getTrackbarPos('high red' ,name)
                h_g=cv2.getTrackbarPos('high green' ,name)
                h_b=cv2.getTrackbarPos('high blue' ,name)
            else:
                l_r = rgb.l_r
                l_g = rgb.l_g
                l_b = rgb.l_b
                h_r = rgb.h_r
                h_g = rgb.h_g
                h_b = rgb.h_b
            key = cv2.waitKey(0)
            # 每1毫秒刷新一次，当输入q键的时候，结束整个主程序
            if key == ord('s'):
                low=np.array([l_b ,l_g ,l_r])
                high=np.array([h_b ,h_g ,h_r])
                color_range = np.vstack((low,high))
                print(color_range,color_range_path)
                np.savetxt(color_range_path,color_range)
                # cv2.putText(None ,'000' ,(50 ,300) ,font ,1.2 ,(255 ,255 ,255) ,2)
            elif key ==ord('q'):
                break
            else:
                print(key)

        cv2.destroyAllWindows()

        # cv2.imwrite(i.replace('.png','.jpg'),img_nopiece)
        # viewImage(gray)