import numpy as np
import pandas as pd
import os
import cv2
import math
import random
import matplotlib.pyplot as plt
import shutil
from sklearn.preprocessing import QuantileTransformer
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

df =pd.read_csv(r"D:\final\CICIDS2017_normalize/XSS.csv")

df.head()


#Generate images corresponding to each class
# df0=df[df['Label']=='BENIGN'].drop(['Label'],axis=1)
# df1=df[df['Label']=='Bot'].drop(['Label'],axis=1)
# df2=df[df['Label']=='BRUTEFORCE'].drop(['Label'],axis=1)
df3=df[df['Label']=='DDoS'].drop(['Label'],axis=1)
df4=df[df['Label']=='DoS GoldenEye'].drop(['Label'],axis=1)
df5=df[df['Label'] == 'DoS Hulk'].drop(['Label'], axis=1)
df6=df[df['Label']=='DoS Slowhttptest'].drop(['Label'],axis=1)
df7=df[df['Label']=='DoS slowloris'].drop(['Label'],axis=1)
df8=df[df['Label']=='FTP-Patator'].drop(['Label'],axis=1)
df9=df[df['Label']=='Heartbleed'].drop(['Label'],axis=1)
df10=df[df['Label']=='Infiltration'].drop(['Label'],axis=1)
df11=df[df['Label']=='PortScan'].drop(['Label'],axis=1)
df12=df[df['Label']=='SQLINJECTION'].drop(['Label'],axis=1)
df13=df[df['Label']=='SSH-Patator'].drop(['Label'],axis=1)
df14=df[df['Label']=='XSS'].drop(['Label'],axis=1)

# ***********Generate 95*95 color images for class 0 (Normal)************
'''
count=0
ims = []

image_path = r"D:\final\image\BENIGN"
os.makedirs(image_path, exist_ok=True)

for i in range(0, len(df0)):
    count=count+1
    if count<=117:
        im=df0.iloc[i].values
        ims=np.append(ims,im)
    else:
        ims=np.array(ims).reshape(39,39,3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(image_path+str(i)+'.png')
        count=0
        ims = []

# ***********Generate 95*95 color images for class 1 (DDoS_UDP)************
count=0
ims = []

image_path = r"D:\final\image\BACKDOOR_MALWARE"
os.makedirs(image_path, exist_ok=True)

for i in range(0, len(df1)):
    count=count+1
    if count<=117:
        im=df1.iloc[i].values
        ims=np.append(ims,im)
    else:
        ims=np.array(ims).reshape(39,39,3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(image_path+str(i)+'.png')
        count=0
        ims = []

# ***********Generate 95*95 color images for class 2 (DDoS_ICMP)************
count=0
ims = []

image_path = r"D:\final\image\DDOS-ICMP_FLOOD"
os.makedirs(image_path, exist_ok=True)

for i in range(0, len(df2)):
    count=count+1
    if count<=117:
        im=df2.iloc[i].values
        ims=np.append(ims,im)
    else:
        ims=np.array(ims).reshape(39,39,3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(image_path+str(i)+'.png')
        count=0
        ims = []
# ***********Generate 95*95 color images for class 3 (SQL_injection )************
count=0
ims = []

image_path = r"D:\final\image\DDOS-SYN_FLOOD"
os.makedirs(image_path, exist_ok=True)

for i in range(0, len(df3)):
    count=count+1
    if count<=117:
        im=df3.iloc[i].values
        ims=np.append(ims,im)
    else:
        ims=np.array(ims).reshape(39,39,3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(image_path+str(i)+'.png')
        count=0
        ims = []

# ***********Generate 95*95 color images for class 4 (DDoS_TCP)************
count=0
ims = []

image_path = r"D:\final\image\DDOS-HTTP-FLOOD"
os.makedirs(image_path, exist_ok=True)

for i in range(0, len(df4)):
    count=count+1
    if count<=117:
        im=df4.iloc[i].values
        ims=np.append(ims,im)
    else:
        ims=np.array(ims).reshape(39,39,3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(image_path+str(i)+'.png')
        count=0
        ims = []


# ***********Generate 95*95 color images for class 4 (DDoS_TCP)************
count=0
ims = []

image_path = r"D:\final\image\VULNERABILITYSCAN"
os.makedirs(image_path, exist_ok=True)

for i in range(0, len(df5)):
    count=count+1
    if count<=117:
        im=df5.iloc[i].values
        ims=np.append(ims,im)
    else:
        ims=np.array(ims).reshape(39,39,3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(image_path+str(i)+'.png')
        count=0
        ims = []

count=0
ims = []

image_path = r"D:\final\image\XSS"
os.makedirs(image_path, exist_ok=True)

for i in range(0, len(df6)):
    count=count+1
    if count<=117:
        im=df6.iloc[i].values
        ims=np.append(ims,im)
    else:
        ims=np.array(ims).reshape(39,39,3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(image_path+str(i)+'.png')
        count=0
        ims = []

count=0
ims = []

image_path = r"D:\final\image/UPLOADING"
os.makedirs(image_path, exist_ok=True)
for i in range(0, len(df7)):
    count=count+1
    if count<=117:
        im=df7.iloc[i].values
        ims=np.append(ims,im)
    else:
        ims=np.array(ims).reshape(39,39,3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(image_path+str(i)+'.png')
        count=0
        ims = []



count=0
ims = []

image_path = r"D:\final\image\SQLINJECTION"
os.makedirs(image_path, exist_ok=True)
for i in range(0, len(df8)):
    count=count+1
    if count<=117:
        im=df8.iloc[i].values
        ims=np.append(ims,im)
    else:
        ims=np.array(ims).reshape(39,39,3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(image_path+str(i)+'.png')
        count=0
       ims = []
'''
count=0
ims = []

image_path = r"D:\final\image_cicids2017\XSS"
os.makedirs(image_path, exist_ok=True)
for i in range(0, len(df14)):
    count=count+1
    if count<=234:
        im=df14.iloc[i].values
        ims=np.append(ims,im)
    else:
        ims=np.array(ims).reshape(78,78,3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(image_path+str(i)+'.png')
        count=0
        ims = []
