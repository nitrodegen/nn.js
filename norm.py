import numpy as np 
import os,io,sys

fd = open("./train.csv","r")
data = fd.read().split("\n")[1:32]
imgs = [ ]

for x in data:
    
    x = x.split(",")
    xd =[] 

    for z in x:
        xd.append(int(z))
    xd =xd[1:]
    print(len(xd))
    imgs.append(xd)
    
imgs = np.array(imgs,dtype=np.float64)[0]
norm = (imgs-np.min(imgs))/(np.max(imgs)-np.min(imgs))
print(np.min(imgs))