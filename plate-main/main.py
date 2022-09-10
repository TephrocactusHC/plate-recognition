# -*- coding = 'utf-8' -*-
'''
#@Time: 2022/7/20 17:40
#@Author: TephrocactusHC
#@File: test121212121.py
#@Project: TA_TransE
#@Software: PyCharm
'''
import cv2,os
import numpy as np
from cv import cvmain
from cnn_predict import get_enu
from cnn_predict_chs import get_chs
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# tf.compat.v1.disable_eager_execution()

WINDOW_TITLE = "Plage Locate"  ##默认展示图片位置
char_w, char_h = 20, 20
# 左右切割
def horizontal_cut_chars(plate):
    char_addr_list = []
    area_left,area_right,char_left,char_right= 0,0,0,0
    img_w = plate.shape[1]

    # 获取车牌每列边缘像素点个数
    def getColSum(img,col):
        sum = 0
        for i in range(img.shape[0]):
            sum += round(img[i,col]/255)
        return sum;

    sum = 0
    for col in range(img_w):
        sum += getColSum(plate,col)
    # 每列边缘像素点必须超过均值的60%才能判断属于字符区域
    col_limit = 0#round(0.5*sum/img_w)
    # 每个字符宽度也进行限制
    charWid_limit = [round(img_w/12),round(img_w/5)]
    is_char_flag = False

    for i in range(img_w):
        colValue = getColSum(plate,i)
        if colValue > col_limit:
            if is_char_flag == False:
                area_right = round((i+char_right)/2)
                area_width = area_right-area_left
                char_width = char_right-char_left
                if (area_width>charWid_limit[0]) and (area_width<charWid_limit[1]):
                    char_addr_list.append((area_left,area_right,char_width))
                char_left = i
                area_left = round((char_left+char_right) / 2)
                is_char_flag = True
        else:
            if is_char_flag == True:
                char_right = i-1
                is_char_flag = False
    # 手动结束最后未完成的字符分割
    if area_right < char_left:
        area_right,char_right = img_w,img_w
        area_width = area_right - area_left
        char_width = char_right - char_left
        if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
            char_addr_list.append((area_left, area_right, char_width))
    return char_addr_list

def get_chars(car_plate):
    img_h,img_w = car_plate.shape[:2]
    h_proj_list = [] # 水平投影长度列表
    h_temp_len,v_temp_len = 0,0
    h_startIndex,h_end_index = 0,0 # 水平投影记索引
    h_proj_limit = [0.2,0.8] # 车牌在水平方向得轮廓长度少于20%或多余80%过滤掉
    char_imgs = []

    # 将二值化的车牌水平投影到Y轴，计算投影后的连续长度，连续投影长度可能不止一段
    h_count = [0 for i in range(img_h)]
    for row in range(img_h):
        temp_cnt = 0
        for col in range(img_w):
            if car_plate[row,col] == 255:
                temp_cnt += 1
        h_count[row] = temp_cnt
        if temp_cnt/img_w<h_proj_limit[0] or temp_cnt/img_w>h_proj_limit[1]:
            if h_temp_len != 0:
                h_end_index = row-1
                h_proj_list.append((h_startIndex,h_end_index))
                h_temp_len = 0
            continue
        if temp_cnt > 0:
            if h_temp_len == 0:
                h_startIndex = row
                h_temp_len = 1
            else:
                h_temp_len += 1
        else:
            if h_temp_len > 0:
                h_end_index = row-1
                h_proj_list.append((h_startIndex,h_end_index))
                h_temp_len = 0

    # 手动结束最后得水平投影长度累加
    if h_temp_len != 0:
        h_end_index = img_h-1
        h_proj_list.append((h_startIndex, h_end_index))
    # 选出最长的投影，该投影长度占整个截取车牌高度的比值必须大于0.5
    h_maxIndex,h_maxHeight = 0,0
    for i,(start,end) in enumerate(h_proj_list):
        if h_maxHeight < (end-start):
            h_maxHeight = (end-start)
            h_maxIndex = i
    if h_maxHeight/img_h < 0.5:
        return char_imgs
    chars_top,chars_bottom = h_proj_list[h_maxIndex][0],h_proj_list[h_maxIndex][1]

    plates = car_plate[chars_top:chars_bottom+1,:]

    char_addr_list = horizontal_cut_chars(plates)

    for i,addr in enumerate(char_addr_list):
        char_img = car_plate[chars_top:chars_bottom+1,addr[0]:addr[1]]
        char_img = cv2.resize(char_img,(char_w,char_h))
        char_imgs.append(char_img)
    return char_imgs

def extract_char(car_plate):
    gray_plate = cv2.cvtColor(car_plate,cv2.COLOR_BGR2GRAY)
    ret,binary_plate = cv2.threshold(gray_plate,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    char_img_list = get_chars(binary_plate)
    return char_img_list


def mainmain(inputpath):



    ###这里需要前端能指定输入的图片地址，把我这个改了
    cvmain(originpath=inputpath)
    ###



    ###这张图片是处理好的车牌，需要展示出来
    img = cv2.imread('messigray.png')
    ###


    char_img_list = extract_char(img)
    broadstr=''
    for i,eachimg in enumerate(char_img_list):
        # cv2.imshow(WINDOW_TITLE, eachimg)
        # cv2.waitKey(0)
        path=str(i)+'dig1.jpg'
        cv2.imwrite(path, eachimg)
        if i==0:
            a=get_chs(path)
            broadstr+=a
        else:
            a=get_enu(path)
            broadstr += a

    ###最后的结果，需要展示出来
    if broadstr!='':
        return broadstr
    else:
        return '没有识别到车牌，请确认照片是否清晰完整！'
    ###
# print(mainmain('train.jpg'))