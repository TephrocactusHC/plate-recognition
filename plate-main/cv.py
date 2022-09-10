# @TODO#暂时没有实现字符的分割。后续增加。另外，由于缺乏人工标注位置的数据集，暂时只能通过肉眼观察以确定模型效果，无法批量观察结果。
# #以上整个文件应该在后续中进行封装，然后作为函数直接调用。
#
#
# '''
# 关于级联分类器的使用
# 关于使用级联分类器的原因是单纯使用HSV和其他二值化办法时，对清晰的图像效果明显，但是对于很多过曝等光线条件不好，以及车牌或镜头有一些泥点等情况，效果就会很不理想。
# 由于本过程处理出来的数据相当于后续分类模型的训练集，为了保证后续分类模型的效果，务必保证本步骤检测非常好的效果，因此选择了一个级联分类器使用。
# 本质上说，这一步其实近似于预训练的理念，即采取了由数据集之外的数据（大样本）所产生的训练结果来辅助我们的模型达到更好的效果。
# 在中科大数据集里，较为容易的文件夹（base）达到了非常良好的效果，不过在对中科大数据集中很多被认为困难的数据上，仍然无法做到准确识别的效果。
# 后续如果分类模型的结果非常不理想（尤其是在base上）可能会考虑采用一些神经网络进行车牌的定位和处理，不过那需要新的人工标注位置的数据集作为训练集，暂时不予采用。
# '''
import cv2, math
import numpy as np
WINDOW_TITLE = "Plage Locate"##默认展示图片位置
watch_cascade = cv2.CascadeClassifier('cascade.xml')


def hist_image(img):
    assert img.ndim==2
    hist = [0 for i in range(256)]
    img_h,img_w = img.shape[0],img.shape[1]

    for row in range(img_h):
        for col in range(img_w):
            hist[img[row,col]] += 1
    p = [hist[n]/(img_w*img_h) for n in range(256)]
    p1 = np.cumsum(p)
    for row in range(img_h):
        for col in range(img_w):
            v = img[row,col]
            img[row,col] = p1[v]*255
    return img

def find_board_area(img):
    assert img.ndim==2
    img_h,img_w = img.shape[0],img.shape[1]
    top,bottom,left,right = 0,img_h,0,img_w
    flag = False
    h_proj = [0 for i in range(img_h)]
    v_proj = [0 for i in range(img_w)]

    for row in range(round(img_h*0.5),round(img_h*0.8),3):
        for col in range(img_w):
            if img[row,col]==255:
                h_proj[row] += 1
        if flag==False and h_proj[row]>12:
            flag = True
            top = row
        if flag==True and row>top+8 and h_proj[row]<12:
            bottom = row
            flag = False

    for col in range(round(img_w*0.3),img_w,1):
        for row in range(top,bottom,1):
            if img[row,col]==255:
                v_proj[col] += 1
        if flag==False and (v_proj[col]>10 or v_proj[col]-v_proj[col-1]>5):
            left = col
            break
    return left,top,120,bottom-top-10



def img_Transform(car_rect,image):
    img_h,img_w = image.shape[:2]
    rect_w,rect_h = car_rect[1][0],car_rect[1][1]
    angle = car_rect[2]

    return_flag = False
    if car_rect[2]==0:
        return_flag = True
    if car_rect[2]==-90 and rect_w<rect_h:
        rect_w, rect_h = rect_h, rect_w
        return_flag = True
    if return_flag:
        car_img = image[int(car_rect[0][1]-rect_h/2):int(car_rect[0][1]+rect_h/2),
                  int(car_rect[0][0]-rect_w/2):int(car_rect[0][0]+rect_w/2)]
        return car_img

    car_rect = (car_rect[0],(rect_w,rect_h),angle)
    box = cv2.boxPoints(car_rect)

    heigth_point = right_point = [0,0]
    left_point = low_point = [car_rect[0][0], car_rect[0][1]]
    for point in box:
        if left_point[0] > point[0]:
            left_point = point
        if low_point[1] > point[1]:
            low_point = point
        if heigth_point[1] < point[1]:
            heigth_point = point
        if right_point[0] < point[0]:
            right_point = point

    if left_point[1] <= right_point[1]:  # 正角度
        new_right_point = [right_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w*2), round(img_h*2)))
        car_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]

    elif left_point[1] > right_point[1]:  # 负角度
        new_left_point = [left_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, M, (round(img_w*2), round(img_h*2)))
        car_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]

    return car_img

def pre_process(orig_img):

    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow(WINDOW_TITLE, gray_img)
    # cv2.waitKey(0)
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    # cv2.imshow(WINDOW_TITLE, blur_img)
    # cv2.waitKey(0)
    sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)
    sobel_img = cv2.convertScaleAbs(sobel_img)
    # cv2.imshow(WINDOW_TITLE, sobel_img)
    # cv2.waitKey(0)
    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    # cv2.imshow(WINDOW_TITLE, hsv_img)
    # cv2.waitKey(0)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    # 黄色色调区间[26，34],蓝色色调区间:[100,124]
    blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
    blue_img = blue_img.astype('float32')
    # cv2.imshow(WINDOW_TITLE, blue_img)
    # cv2.waitKey(0)
    mix_img = np.multiply(sobel_img, blue_img)

    mix_img = mix_img.astype(np.uint8)

    ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #去除白点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
    #膨胀，腐蚀
    close_img=cv2.dilate(close_img,kernelx)
    close_img=cv2.erode(close_img,kernelx)
    close_img = cv2.dilate(close_img, kernely)
    close_img = cv2.erode(close_img, kernely)
    #中值滤波去除噪点
    close_img = cv2.medianBlur(close_img, 15)

    # cv2.imshow('close', close_img)
    # cv2.waitKey()
    return close_img
def verify_scale(rotate_rect):
   error = 0.4
   min_area = 80*80
   max_area = 36 * 40 * 9 * 40
   min_aspect = 3*(1-error)
   max_aspect = 3*(1+error)
   theta = 30
   # 宽或高为0，不满足矩形直接返回False
   if rotate_rect[1][0]==0 or rotate_rect[1][1]==0:
       return False

   r = rotate_rect[1][0]/rotate_rect[1][1]
   r = max(r,1/r)
   area = rotate_rect[1][0]*rotate_rect[1][1]
   if area>min_area and area<max_area and r>min_aspect and r<max_aspect:
       return True
   return False
# 车牌定位+矫正
def locate_carPlate(orig_img,pred_image):
    temp1_orig_img = orig_img.copy() #调试用
    temp2_orig_img = orig_img.copy()  # 调试用
    temp3_orig_img = orig_img.copy()  # 调试用
    contours,heriachy = cv2.findContours(pred_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        # 获取轮廓最小外接矩形，返回值rotate_rect
        rotate_rect=cv2.minAreaRect(contour)
        #做一个判断
        if verify_scale(rotate_rect):
            cv2.drawContours(temp1_orig_img, contours, i, (0, 255, 255), 2)
            car_plate = img_Transform(rotate_rect, temp2_orig_img)
            return car_plate
    return temp3_orig_img
    #         cnt=contours[i]
    #         h,w=temp1_orig_img.shape[:2]
    #         [vx,vy,x,y]=cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
    #         k=vy/vx
    #         b=y-k*x
    #         lefty=b
    #         righty=k*w+b
    #         a=math.atan(k)
    #         a=math.degrees(a)
    #         h2,w2=temp3_orig_img.shape[:2]
    #         img=cv2.line(temp2_orig_img,(w,int(righty)),(0,int(lefty)),(0,255,0),2)
    #         M=cv2.getRotationMatrix2D((w/2,h/2),a,0.8)#旋转中心，旋转角度，缩放比例
    #         dst=cv2.warpAffine(temp3_orig_img,M,(int(w*1.1),int(h*1.1)))
    #         dst = cv2.GaussianBlur(dst, (5, 5), 0)  ##高斯除噪
    #         rect=cv2.boundingRect(contour)
    #         x=rect[0]
    #         y = rect[1]
    #         weight = rect[2]
    #         height = rect[3]
    #         det=dst[y-10:y+10+height,x-5:x+5+weight]
    #         temp_det=det.copy()
    #         try:
    #             if get_broad_result(det).all()!=None:
    #                 return get_broad_result(temp_det)
    #         except AttributeError:
    #             pass
    # try:
    #     if get_broad_result1(dst).all()==None:
    #         return dst
    #     else:
    #         return get_broad_result1(dst)
    # except AttributeError:
    #     if get_broad_result1(dst).all() == None:
    #         return dst
    #     else:
    #         return get_broad_result1(dst)
    # except UnboundLocalError:
    #     return temp4_orig_img


def get_broad_result(image):
    resize_h = 100
    image = cv2.resize(image, (400, resize_h))
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_tmp=image.copy()
    watches = watch_cascade.detectMultiScale(image_gray, 1.1, 2, minSize=(300, 70), maxSize=(400, 100))
    x1,y1,w1,h1=0,0,400,100
    for (x, y, w, h) in watches:
        cv2.rectangle(image_tmp, (x-10, y), (x + w+30, y + h), (0, 0, 255), 3)
        y1=int(y)
        h1=int(h)
        if x-10<0 :
            x1=0
        else:
            x1=x-10

        if x+w+30>400:
            w1=w+x
        else:
            w1=x+w+30
    image=image[y1:y1+h1,x1:w1]
    # cv2.imshow("image", image_tmp)
    # cv2.waitKey(0)
    cut_gray = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(cut_gray, (5, 5), 0)  ##高斯除噪
    # th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    ret,th=cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    return 255-th

# def get_broad_result1(image):
#     resize_h = 1000
#     height = image.shape[0]
#     scale = image.shape[1] / float(image.shape[0])
#     image = cv2.resize(image, (int(scale * resize_h), resize_h))
#     image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     watches = watch_cascade.detectMultiScale(image_gray, 1.1, 2, minSize=(36, 9), maxSize=(36*60, 9*60))
#     # print("1检测到车牌数", len(watches))
#     for (x, y, w, h) in watches:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)  ##先确定车牌的大致范围
#         cut_img = image[y:y + h, x - 15:x + w]  ##然后做具体的裁剪
#         # 函数的裁剪坐标为[y0:y1, x0:x1]
#         cut_gray = cv2.cvtColor(cut_img, cv2.COLOR_RGB2GRAY)  ##处理成灰度图
#         # cv2.imshow(WINDOW_TITLE, cut_gray)
#         # cv2.waitKey()
#         ###最后展示处理好的车牌效果
#         blur = cv2.GaussianBlur(cut_gray, (5, 5), 0)  ##高斯除噪
#         th=cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
#
#         # cv2.imshow(WINDOW_TITLE, th)
#         # cv2.waitKey()
#         return th


def cvmain(originpath):
    car_plate_w,car_plate_h = 136,36
    char_w,char_h = 20,20
    img = cv2.imread(originpath)
    # 预处理
    pred_img = pre_process(img)

    # 新的车牌图片
    new_car_img = locate_carPlate(img,pred_img)
    # cv2.imshow(WINDOW_TITLE, new_car_img)
    # cv2.waitKey(0)
    #处理好的车牌
    new_car_img1=get_broad_result(new_car_img)
    # cv2.imshow(WINDOW_TITLE, new_car_img1)
    # cv2.waitKey(0)
    cv2.imwrite('messigray.png', new_car_img1)







