import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import model_from_json
import csv
from PIL import Image, ImageEnhance
import glob, os, PIL.ImageOps

json_file = open('model_w.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_w.h5")

def rectifier(arr):
    l,b=arr.shape
    for i in range(l):
        for j in range(b):
            if arr[i][j]<253:
                arr[i][j]=0
            else :
                arr[i][j]=255
    return arr

def dot_remove(tt):
    l,b=tt.shape
    for i in range(l):
        for j in range(b):
            if tt[i][j]!=0:
                c1,c2,c3,c4=(max(0,i-15),min(l-1,i+15),max(0,j-15),min(b-1,j+15))
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

#def localize(arr):
#    l,b=arr.shape
#    ver_lim=[]
#    alt=255
#    for j in range(b):
#    flag=0
#        for i in range(l):
#            if arr[i][j]==alt:
#                ver_lim.append(j)
#                flag=1
#                break
#        if flag==1:
#            if alt==255:
#                alt=0
#            else:
#                alt=255

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
