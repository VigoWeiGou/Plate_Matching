# conding=utf-8


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle



def imshow(win_name, img):
    if len(img.shape)==3:
        (H, W, chn)=img.shape
    else:
        (H, W) = img.shape

    while ((H>712 )| (W>1024)):
        H=int(H/2)
        W=int(W/2)

    cv2.namedWindow(win_name, cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(win_name, W, H)
    cv2.imshow(win_name, img)


def get_all_counters(img_bgr):
    if(len(img_bgr.shape) == 2):
        img_gray = img_bgr
    else:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    T = 30;
    ret, img_thresh = cv2.threshold(img_gray, T, 255, cv2.THRESH_BINARY)
    imshow("Thresh", img_thresh)
    img, cnts, hierarchy = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cnts

def get_ref_contour(img_bgr):
    cnts = get_all_counters(img_bgr)

    for cnt in cnts:
        pass
    pass



def create_templete(img):
    cnts = get_all_counters(img)
    print("len(img_cnts): ", len(cnts))

    cnts_dict = cnts2dict(cnts)

    with open("cnts.pickle", 'wb') as fw:
        pickle.dump(cnts_dict, fw)

    print(cnts_dict)



def cnts2dict(cnts):
    cnts_dict={}
    for i, cnt in enumerate(cnts):
        cnts_dict[i]=cnt
    return  cnts_dict

def detect(img_ref, img):
    # cv2.namedWindow("ref_img", cv2.WINDOW_FREERATIO)
    # cv2.namedWindow("img", cv2.WINDOW_FREERATIO)

    imshow("Ref image", img_ref)
    cnts = get_all_counters(img)

    with open("./cnts.pickle", 'rb') as fr:
        cnts_dict = pickle.load(fr)
        ref_cnts = list(cnts_dict.values())

    for cnt in cnts:

        M = cv2.moments(cnt)
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))

        closest_k = None
        min_dist = None

        for k, ref_cnt in enumerate(ref_cnts):
            ref_M = cv2.moments(ref_cnt)
            ref_cX = int((ref_M["m10"] / ref_M["m00"]))
            ref_cY = int((ref_M["m01"] / ref_M["m00"]))

            # color2 = (0,255,255)
            color2 = (255, 255, 2)
            cv2.putText(img_ref, str(k), (ref_cX, ref_cY), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                        color=color2, thickness=5)

            ret = cv2.matchShapes(ref_cnt, cnt, cv2.CONTOURS_MATCH_I3, 0.0)
            print("Counter %d : %f" %(k, ret))
            if min_dist is None or min_dist>ret:
                min_dist = ret
                closest_k = k

        # color1 = (0,255,127)
        color1 = (2,25,254)
        cv2.putText(img, str(closest_k), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color1, thickness=5)



    imshow("ref_img", img_ref)
    imshow("img", img)




if __name__=='__main__':
    ref_path = "test1.jpg"
    img_ref = cv2.imread(ref_path)
    create_templete(img_ref)
    img_path = "test2.jpg"
    img = cv2.imread(img_path)

    detect(img_ref, img)


    cv2.waitKey(0)










