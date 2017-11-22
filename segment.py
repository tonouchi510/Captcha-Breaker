import cv2
import CaptchaBreaker_v2 as cb
import numpy as np

# 単純なスライディングウィンドウ方式による切り出し
def sliding_window(img):
    list = []
    i = 0
    while ((i+100) < 214):
        seg_im = (img[0:100, i:i+100])
        y, index = cb.inference(seg_im)
        if (y[index] > 0.3):
            list.append(index+65)
            print(list)
            i += 20
        else:
            i += 5
    return list


# 2値画像の領域分割による方式
def segment(img):
    # 画像の白黒を反転
    imgv = ~img
    # EXTERNALで最外輪郭のみ抽出
    imgv, contours, hierarchy = cv2.findContours(imgv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    num = len(contours)

    # 輪郭領域を左から順に扱えるようにする
    index = num*[0]
    tmpls = num*[0]
    for i in range(num):
        size = len(contours[i])
        contour = contours[i].reshape(size, 2)
        _, minPx = minPoint(contour)
        tmpls[i] = minPx
    for i in range(num):
        tmp = min(tmpls)
        index[i] = tmpls.index(tmp)
        tmpls[index[i]] = 215

    result = []
    for i in index:
        # contoursの配列の形を整形
        size = len(contours[i])
        contour = contours[i].reshape(size, 2)
        # 各輪郭の最大最小座標を求める
        minPy, minPx = minPoint(contour)
        maxPy, maxPx = maxPoint(contour)
        dx = maxPx - minPx
        # 抽出した輪郭の大きさで切り出す位置(start)を調整
        if dx >= 100:
            start = minPx
        elif dx < 100:
            tmps = max( int(minPx - (100-dx)/2), 0)
            start = min(tmps, 114)

        tmp = img
        cv2.fillPoly(tmp, [contour], 0)
        for j in range(num):
            if j != i:
               cv2.fillPoly(tmp, [contours[j]], 255)
        seg_img = (tmp[0:100, start:start+100])
        area = 10000 - cv2.countNonZero(seg_img)
        # 領域が小さいものは除く
        if area < 150:
            continue
        ''' デバッグ時用(各予測時の画像を表示)
        cv2.imshow('image', seg_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        # CNNによる予測
        x = seg_img / 255.0
        out, id = cb.inference(x)
        y = chr(id+65) # CNNの出力をaskii表現にそろえる
        #print(y)
        result.append(y)

    return result

def lr(contour, size):
    right = np.zeros(100)
    left = np.zeros(100) + 215
    for i in range(size):
        if right[contour[i][1]] < contour[i][0]:
            right[contour[i][1]] = contour[i][0]
        if left[contour[i][1]] > contour[i][0]:
            left[contour[i][1]] = contour[i][0]
    return left, right

def minPoint(contour):
    # 最小座標を求める
    minx = contour[0][0]
    miny = contour[0][1]

    for i in range(1,len(contour)):
        if (minx > contour[i][0]):
            minx = contour[i][0]
        if (miny > contour[i][1]):
            miny = contour[i][1]
    return miny, minx

def maxPoint(contour):
    # 最大座標を求める
    maxx = contour[0][0]
    maxy = contour[0][1]
    for i in range(1,len(contour)):
        if (maxx < contour[i][0]):
            maxx = contour[i][0]
        if (maxy < contour[i][1]):
            maxy = contour[i][1]
    return maxy, maxx

