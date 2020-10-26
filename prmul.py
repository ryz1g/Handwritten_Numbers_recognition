import helping_functions as hf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
import PIL.Image

s="test1.jpg"
img=Image.open(s)
#img.show()
img1=hf.preprocess(img)
tt=np.array(img1)
tt=hf.rectifier(tt)
#print(hf.output(tt))
#input()
ll=hf.local(tt)
img=img.crop((ll[0], ll[1], ll[3], ll[2]))
img1=hf.preprocess(img)
tt=hf.rectifier(np.array(img1))
lo=hf.split(tt)
li=[]
li.append(0)
li.append(lo[0])
for i in range(len(lo)-1):
    if (lo[i]+1)!=lo[i+1]:
        li.append(lo[i+1])
nums=[]
l,b=tt.shape
plt.imshow(tt,cmap="gray")
plt.show()
li.append(b)
for i in range(len(li)-1):
    ii=img.crop((li[i],0,li[i+1],l))
    ii=hf.preprocess(ii)
    ii=hf.rectifier(np.array(ii))

    trr=hf.local(ii)
    ii=PIL.Image.fromarray(ii)
    try:
        ii=ii.crop((trr[0],trr[1],trr[3],trr[2]))
    except:
        _=1
    ii=hf.preprocess(ii)
    ii=hf.rectifier(np.array(ii))
    #

    ii=PIL.Image.fromarray(ii)
    ii=hf.preprocess(ii)
    b1,l1=ii.size
    ch=l1-b1
    if ch>0:            #making square
        ii=ii.crop((int(ch/2)*-1,0,b1+int(ch/2),l1))
    else:
        ii=ii.crop((0,int(ch/2),b1,l1+(-1*int(ch/2))))
    ii=ii.resize((24,24))
    #ii=ii.crop((max(-0.1*b1,-25),-0.1*l1,min(1.1*b1,b1+25),1.1*l1))         #expanding horizons
    ii=ii.crop((-2,-2,26,26))

    enhancer=ImageEnhance.Contrast(ii)
    ii=enhancer.enhance(4)
    it=hf.rectifier(np.array(ii))

    plt.imshow(it, cmap="gray")
    plt.show()
    c=0
    for i in range(28):
        for j in range(28):
            if it[i][i]>250:
                c=c+1
    if c>20:
        nums.append(it)

for i in nums:
    print(hf.output(i))
