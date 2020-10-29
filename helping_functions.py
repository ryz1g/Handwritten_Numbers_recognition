import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageEnhance
import PIL.ImageOps
import PIL
import cv2

json_file = open('model_w.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_w.h5")

def edge_detect(arr):
    #edge detection
    fil1=np.array([[0,-1,0],
                    [-1,4,-1],
                  [0,-1,0]])
    #blurring
    fil2=np.array([[1,2,1],
                    [2,4,2],
                  [1,2,1]])/16
    #sharpening
    fil3=np.array([[0,-1,0],
                  [-1,5,-1],
                  [0,-1,0]])
    arr=cv2.filter2D(arr, -1, fil2)
    arr=cv2.filter2D(arr, -1, fil2)
    arr=cv2.Canny(arr, 112, 479, 20.0)
    return arr


def new_rectifier(arr):
    l,b=(84,479)
    img=PIL.ImageOps.invert(PIL.Image.fromarray(arr))
    enhancer=ImageEnhance.Contrast(img)
    img=enhancer.enhance(10)
    img=img.crop((0,184,479,296))
    arr=np.array(img)
    arr=edge_detect(arr)
    return arr

def rectifier(arr):
    l,b=arr.shape
    for i in range(l):
        for j in range(b):
            if arr[i][j]<50:
                arr[i][j]=0
            else :
                arr[i][j]=255
    return arr

def dot_remove(tt):
    l,b=tt.shape
    for i in range(l):
        for j in range(b):
            if tt[i][j]!=0:
                c1,c2,c3,c4=(max(0,i-3),min(l-1,i+3),max(0,j-3),min(b-1,j+3))
                flag=0
                for loo in range(c3,c4):
                    if tt[c1][loo]!=0 and c1!=i and loo!=j:
                        flag=1
                        break
                for loo in range(c3,c4):
                    if tt[c2][loo]!=0 and c2!=i and loo!=j:
                        flag=1
                        break
                for loo in range(c1,c2):
                    if tt[loo][c3]!=0 and loo!=i and c3!=j:
                        flag=1
                        break
                for loo in range(c1,c2):
                    if tt[loo][c4]!=0 and loo!=i and c4!=j:
                        flag=1
                        break
                if flag==0:
                    tt[i][j]=0
        return tt


def preprocess(img):
    img=img.convert(mode="L")
    #img=img.resize((28,28))
    img=PIL.ImageOps.invert(img)
    enhancer=ImageEnhance.Contrast(img)
    img=enhancer.enhance(4)
    return img

def output(arr):
    return loaded_model.predict_classes(arr.reshape(1,28,28,1).astype("float32"))

def output_multiple(img):
    li=[]
    nums=[]
    try:
        ll=local(np.array(img))
        img=img.crop((ll[0], ll[1], ll[3], ll[2]))
        ll=split(np.array(img))
        li.append(0)
        li.append(ll[0])
        for i in range(len(ll)-1):
            if (ll[i]+1)!=ll[i+1]:
                li.append(ll[i+1])

        tt=np.array(img)
        l,b=tt.shape
        li.append(b)
    except:
        _=1
    #plt.imshow(tt,cmap="gray")
    #plt.show()
    for i in range(len(li)-1):
        try :
            ii=img.crop((li[i],0,li[i+1],l))
            trr=local(np.array(ii))
            try:
                ii=ii.crop((trr[0],trr[1],trr[3],trr[2]))
            except:
                _=1
            b1,l1=ii.size
            ch=l1-b1
            if ch>0:            #making square
                ii=ii.crop((int(ch/2)*-1,0,b1+int(ch/2),l1))
            else:
                ii=ii.crop((0,int(ch/2),b1,l1+(-1*int(ch/2))))
            ii=ii.resize((24,24))
            #ii=ii.crop((max(-0.1*b1,-25),-0.1*l1,min(1.1*b1,b1+25),1.1*l1))         #expanding horizons
            ii=ii.crop((-2,-2,26,26))
            it=np.array(ii)
            it=rectifier(it)
            c=0
            for i in range(28):
                for j in range(28):
                    if it[i][i]>250:
                        c=c+1
            if c>20:
                nums.append(it)
        except:
            _=1
    opp=0
    for i in nums:
        opp=opp*10+output(i)[0]
    print(opp)
    return opp

def split(arr):
    l,b=arr.shape
    ver_lim=[]
    flag=0
    for j in range(b):
        flag=0
        for i in range(l):
            if arr[i][j]!=0:
                flag=1
                break
        if flag==0:
            ver_lim.append(j)
    return ver_lim


def local(arr):
    l,b=arr.shape
    lims=[]
    alt=200
    flag=0
    for j in range(b):
        for i in range(l):
            if arr[i][j]>=alt:
                lims.append(j)
                flag=1
                break
        if flag==1:
            break
    flag=0
    for i in range(l):
        for j in range(b):
            if arr[i][j]>=alt:
                lims.append(i)
                flag=1
                break
        if flag==1:
            break
    flag=0
    for i in range(l):
        for j in range(b):
            if arr[l-i-1][b-j-1]>=alt:
                lims.append(l-i-1)
                flag=1
                break
        if flag==1:
            break
    flag=0
    for j in range(b):
        for i in range(l):
            if arr[l-i-1][b-j-1]>=alt:
                lims.append(b-j-1)
                flag=1
                break
        if flag==1:
            break
    return lims
