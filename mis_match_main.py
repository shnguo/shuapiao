# -*- coding: utf-8 -*-
import sys
import os
import datetime
import config
import cv2
import urllib
import requests
import time
import numpy as np
import multiprocessing
from multiprocessing import Pool, Queue, Process
from multiprocessing.dummy import Pool as ThreadPool
from mis_match_base import get_train_set, get_test_wrong_set, get_phash
from gs_log import getLogger
from pprint import pprint
from scipy import stats
from requests.adapters import HTTPAdapter
import urllib3
urllib3.disable_warnings()
# import matplotlib
# matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
try:
    import cPickle as pickle
except:
    import Pickle as pickle

__author__ = 'shengguo'

logger = getLogger(__name__)
base_path = os.path.dirname(os.path.realpath(__file__))
# s = requests.session()
# s.keep_alive = False
s = requests.Session()
s.mount('http://i.ebayimg.com/', HTTPAdapter(max_retries=0))
s.mount('http://thumbs.ebaystatic.com/', HTTPAdapter(max_retries=0))

def fix_url(url):
    if '?' in url:
        url = url.split('?')[0]

    if url.endswith('_50x50.jpg'):
        url = url[0:len(url) - len('_50x50.jpg')]
    #220x220xz.jpgjpg
    if url.endswith('.220x220xz.jpgjpg'):
        url = url[0:len(url) - len('.220x220xz.jpgjpg')] + '.jpg'

    if url.endswith('.220x220.jpgjpg'):
        url = url[0:len(url) - len('.220x220.jpgjpg')] + '.jpg'

    return url

def url_to_image(url):
    
    try:
        if 'i.ebayimg.com' in url or 'thumbs.ebaystatic.com' in url:
            return None
        url = fix_url(url)
        r=s.get(url, timeout=3600, verify=False)
        if r.status_code == requests.codes.ok:
            image = np.asarray(bytearray(r.content), dtype="uint8")

            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        # image = cv2.imdecode(image, cv2.COLOR_BGR2GRAY)
        
            return image
        else:
            return None
    except Exception,e:
        logger.error(e)
        return None

def af_compute(des, eps=1e-7):
    desc = des
    desc /= (desc.sum(axis=1, keepdims=True) + eps)
    desc = np.sqrt(desc)
    return desc

def calculate_kp(item):
    try:
        c_image_url = item['c_image_url']
        b_image_url = item['b_image_url']
        category_name = item['category_name']
        c_platform = item['c_platform']

        c_image = url_to_image(c_image_url)  
        b_image = url_to_image(b_image_url)

        if c_image is None or b_image is None:
            # print 'b'
            return None

        cphash = get_phash(c_image)
        bphash = get_phash(b_image)
        
        sift = cv2.xfeatures2d.SIFT_create()
        
        kp1, des1 = sift.detectAndCompute(c_image,None)
        
        kp2, des2 = sift.detectAndCompute(b_image,None)

        if len(kp1)==0 or len(kp2)==0:
            return None

        des1 = af_compute(des1)
        des2 = af_compute(des2)

        bf = cv2.BFMatcher()

        if len(des2)<len(des1):
            des1, des2 = des2, des1
        
        matches = bf.knnMatch(des1,des2, k=2)
        
        good = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append([m])
        # rate = (1.0*len(good))/(min(len(des1), len(des2))+1)
        # logger.info(item['c_image_url'])
        if len(good)==0:
            return None
        return len(good), len(kp1), len(kp2), cphash, bphash
    except Exception, e:
        logger.error(e)
        return None
        


if __name__=='__main__':
    if len(sys.argv)<3:
        print 'input platform and category.'
        exit()
    todaystr = time.strftime("%Y-%m-%d")
    platform = sys.argv[1]
    category = sys.argv[2]
    data_conn = config.get_image_match_conn()
    data_cursor = data_conn.cursor(buffered=True)
    # test_set = get_test_set(data_cursor)
    limit_count = 100
    # train_set = get_train_set(data_cursor, platform, category, limit_count)
    # w_limit_count = int(limit_count-int(limit_count*0.9))
    w_limit_count = 5000
    test_wrong_set = get_test_wrong_set(data_cursor, platform, category, w_limit_count)
    test_right_set = get_train_set(data_cursor, platform, category, w_limit_count)
    # mid = int(len(train_set)*0.9)
    # test_right_set = train_set[mid:]
    # train_set = train_set[:mid]
    # cpus = multiprocessing.cpu_count()
    # url = test_set[0]['c_image_url']
    # pic = url_to_image(url)
    # plt.imshow(pic),plt.show()
    # pool = ThreadPool()
    # train_result = map(calculate_kp, train_set)
    test_right_result = map(calculate_kp, test_right_set)
    test_wrong_result = map(calculate_kp, test_wrong_set)
    # pool.close()
    # pool.join()
   
    # train_result = filter(lambda x: x is not None, train_result)
    test_right_result = filter(lambda x: x is not None, test_right_result)
    test_wrong_result = filter(lambda x: x is not None, test_wrong_result)

    # dump1 = base_path+'/train_history/'+todaystr+'_'+platform+'_'+category+'_'+'train_result.dump'
    # pickle.dump(train_result, open(dump1, 'wb'), True)
    dump2 = base_path+'/train_history/'+todaystr+'_'+platform+'_'+category+'_'+'test_right_result.dump'
    pickle.dump(test_right_result, open(dump2, 'wb'), True)
    dump3 = base_path+'/train_history/'+todaystr+'_'+platform+'_'+category+'_'+'test_wrong_result.dump'
    pickle.dump(test_wrong_result, open(dump3, 'wb'), True)

    # train_length = len(train_result)
    # mingood = min([x[0] for x in train_result])
    # vargood = np.var([x[0] for x in train_result])
    # meangood = np.mean([x[0] for x in train_result])
    # SE_good = stats.sem([x[0] for x in train_result],ddof=0)
    # minrate = min([x[3] for x in train_result])
    # varrate = np.var([x[3] for x in train_result])
    # meanrate = np.mean([x[3] for x in train_result])
    # SE_rate = stats.sem([x[3] for x in train_result],ddof=0)
    # right_is_right = filter(lambda x: x[0]>= mingood or x[3]>=minrate, test_right_result)
    # wrong_is_right = filter(lambda x: x[0]>= mingood or x[3]>=minrate, test_wrong_result)
    # TP = len(right_is_right)*1.0
    # FN = (len(test_right_result)-TP)*1.0
    # FP = len(wrong_is_right)*1.0
    # TN = (len(test_wrong_result)-FP)*1.0
    # TPR = TP/(TP+FN)
    # FNR = FN/(TP+FN)
    # FPR = FP/(FP+TN)
    # TNR = TN/(FP+TN)
    # accuracy = (TP+TN)/(TP+FN+FP+TN)
    # average_accuracy = (TP/(TP+FN)+TN/(TN+FP))/2
    # precison = TP/(TP+FP)
    # Recall = TP/(TP+FN)
    # F1 = 2*precison*Recall/(precison+Recall)

    # text = '''
    # ======================== {} : {} =========================
    # train_length = {}
    # mingood = {}
    # vargood = {}
    # meangood = {}
    # SE_good = {}
    # minrate = {}
    # varrate = {}
    # meanrate = {}
    # SE_rate = {}
    # TP = {}
    # FN = {}
    # FP = {}
    # TN = {}
    # TPR = {}
    # FNR = {}
    # FPR = {}
    # TNR = {}
    # accuracy = {}
    # average_accuracy = {}
    # precison = {}
    # Recall = {}
    # F1 = {}
    # ============================================================================
    # '''.format(platform, category, train_length, mingood, vargood, meangood,
    #     SE_good, minrate, varrate,meanrate,SE_rate,
    #     TP,FN,FP,TN,TPR,FNR,FPR,TNR,accuracy,average_accuracy, 
    #     precison,Recall,F1)
    # logger.info(text)

    # data_cursor.close()
    # data_conn.close()


    # pprint(result)
    # q = multiprocessing.Queue()
    # split_set = dict()
    # workers = list()
    # for i in range(1):
    #     # print i
    #     split_set[i]=test_set[i::cpus]
    #     # print split_set[i]
    #     sift = cv2.xfeatures2d.SIFT_create()
    #     bf = cv2.BFMatcher()
    #     p = Process(target=calculate_kp, args=(split_set[i], q, sift, bf))
    #     workers.append(p)
    # for w in workers:
    #     w.start()
    # for w in workers:
    #     w.join()
    # pool = Pool()
    # for i in range(cpus):
    #     split_set[i]=test_set[i::cpus]
    #     sift = cv2.xfeatures2d.SIFT_create()
    #     bf = cv2.BFMatcher()
    #     pool.apply_async(calculate_kp, args=(split_set[i], q, sift, bf))

    # while not q.empty():
    #     pprint(q.get())
    
