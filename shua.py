# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib
import time
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
s = requests.Session()


def url_to_image(url):
    r = s.get(url, timeout=3600, verify=False)
    print r.status_code
    if r.status_code == 200:
        image = np.asarray(bytearray(r.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.COLOR_BGR2GRAY)
        return image


def main():
    while True:
        url = 'http://jcxmt2017.jcrb.com/voteAPI/CodeKaptcha'

        #     r1 = s.get(url)
        #     i = Image.open(BytesIO(r1.content))
        #     i.save('temp111.jpg')
        #     im = Image.open(path)
        #     plt.imshow(im)
        #  plt.show()
        # plt.imshow(i)
        # plt.show()

        # print image
        image = url_to_image(url)
        cv2.imshow('img_thresh', image)
        cv2.waitKey(5000)
        yanzhengma = raw_input()
        # voteString = ''
        # for _ in range(10):
        voteString = '1,1,1,3,3,3,9,9,9,'
        serverUrl = "http://jcxmt2017.jcrb.com/voteAPI/xmtwb/pcVoteAdd" + "?captcha=" + yanzhengma + "&voteString=" + voteString + "&jsoncallback=?"
        jieguo = 'http://jcxmt2017.jcrb.com/voteAPI' + "/result" + "?keyString=" + 'xmtwb' + "&jsoncallback=?"
        index = 1
        while 1:
            print index
            r = s.get(serverUrl)
            # r2 = s.get(serverUrl)
            print r.text
            index+=1
        print 'over'
        # rr = s.get(jieguo)
        # print rr.text


if __name__ == '__main__':
    main()
