# -*- coding: utf-8 -*-
'''
@author: Julien LANGLOIS
'''

import numpy as np
import cv2
import os, sys
import pickle 
from random import shuffle
import radon 
import euler
DATASET_PATH='/home/akatosh/DATASETS'
MODEL_PATH='MULTITUDE/BREATHER'

training=int(sys.argv[1])
validation=int(sys.argv[2])
test=int(sys.argv[3])
shape=sys.argv[4]
size=int(sys.argv[5])
depth=int(sys.argv[6])

def crop(img,groundTruth,delta=5):
    groundTruth=np.array(groundTruth[:],np.int)
    groundTruth=[img.shape[0]-groundTruth[2],img.shape[0]-groundTruth[3],groundTruth[1],groundTruth[0]]
    dX=groundTruth[1]-groundTruth[0]
    dY=groundTruth[3]-groundTruth[2]
    d=np.maximum(dX,dY)+delta
    cX=int(np.sum(groundTruth[0:2])/2)
    cY=int(np.sum(groundTruth[2:4])/2)
    img=img[int(cX-d/2):int(cX+d/2),int(cY-d/2):int(cY+d/2)]
    return img

if shape=='2d':
    
    SET_PATH='SAMPLES_2D'
    PATH=DATASET_PATH+'/'+MODEL_PATH+'/'+SET_PATH
    
    def getFiles(typeOfSet):
        print("Detection des fichiers...")
        files=os.listdir(PATH+'/'+typeOfSet)
        data=[]
        for file in files:
            if file.find('.png')>0 and file.find('output')>=0:
                label=file.split('.png')[0]
                label=[np.float(label.split('_')[1]),np.float(label.split('_')[2])]
                data.append([file,label])
        data=sorted(data, key=lambda x : x[1])
        shuffle(data)
        print("Fichiers detectes et etiquetes.")
        return data
    
    def getDataFromFiles(files,size,typeOfSet):
        print("Chargement des images...")
        data=[]
        av=5
        
        f=open(PATH+'/'+typeOfSet+'/'+'groundTruth.data','rb')
        groundTruth=pickle.load(f)
        f.close()
        
        for i,file in enumerate(files):
            if depth==1:
                delta=5
                for index,gT in enumerate(groundTruth):
                    if gT[0]==file[0]:
                        imgT=cv2.imread(PATH+'/'+typeOfSet+'/'+file[0],0)
                        width=int(imgT.shape[1]-groundTruth[index][4])+delta-(int(imgT.shape[1]-groundTruth[index][3])-delta)
                        height=int(groundTruth[index][1])+delta-(int(groundTruth[index][2])-delta)
                        cX=int((int(imgT.shape[1]-groundTruth[index][4])+delta+(int(imgT.shape[1]-groundTruth[index][3])-delta))/2)
                        cY=int((int(groundTruth[index][1])+delta+(int(groundTruth[index][2])-delta))/2)
                        sizeT=np.maximum(width,height)
                        if sizeT<size:
                            sizeT=size
                        [sX,eX,sY,eY]=[int(cX-sizeT/2),int(cX+sizeT/2),int(cY-sizeT/2),int(cY+sizeT/2)]
                        img=np.zeros([2,size,size])
                        imgT=imgT[sX:eX,sY:eY]
                        if sizeT>size:
                            imgT=cv2.resize(imgT,(size,size))
                        img[0,:,:]=imgT
                        imgT=cv2.imread(PATH+'/'+typeOfSet+'/'+file[0].split('.png')[0]+'.exr',0)*255
                        imgT=imgT[sX:eX,sY:eY]
                        imgT=cv2.resize(imgT,(size,size))
                        img[1,:,:]=imgT
                        #cv2.imshow('test',np.matrix(img[0,:,:],np.uint8))
                        #cv2.waitKey()
            else:
                img=np.zeros([1,size,size])
                imgT=cv2.imread(PATH+'/'+typeOfSet+'/'+file[0],0)
                imgT=cv2.resize(imgT,(size,size))
                img[0,:,:]=imgT
            data.append([img,file[1]])
            if int(100.0*i/len(files))>=av:
                print(str(int(100.0*i/len(files))) + '%')
                av+=5
        print("Images chargees.")
        return data
    
    def exportData(data,typeOfSet):
        print("Creation du dictionnaire...")
        dictionnary=dict()
        dictionnary['data']=[]
        dictionnary['label']=[]
        for d in data:
            
            dictionnary['data'].append(d[0])
            angle1=d[1][0]
            angle2=d[1][1]
            dictionnary['label'].append([np.cos(np.pi*angle1/180.0),
                                         np.sin(np.pi*angle1/180.0),
                                         np.cos(np.pi*angle2/180.0),
                                         np.sin(np.pi*angle2/180.0)])
        pickle.dump(dictionnary, open(PATH+'/'+typeOfSet+'.data', "wb" ) )
        print("Dictionnaire cree.")
    
    if training==1:
        files=getFiles('TRAIN')
        data=getDataFromFiles(files,size,'TRAIN')
        exportData(data,'TRAIN')
    if validation==1:
        files=getFiles('VALIDATION')
        data=getDataFromFiles(files,size,'VALIDATION')
        exportData(data,'VALIDATION')    
    if test==1:
        files=getFiles('TEST')
        data=getDataFromFiles(files,size,'TEST')
        exportData(data,'TEST')
 
if shape=='3d':
    
    SET_PATH='SAMPLES_3D'
    PATH=DATASET_PATH+'/'+MODEL_PATH+'/'+SET_PATH
    
    def getFiles(typeOfSet):
        print("Detection des fichiers...")
        files=os.listdir(PATH+'/'+typeOfSet)
        data=[]
        for file in files:
            if file.find('.png')>0 and file.find('output')>=0:
                label=file.split('.png')[0]
                label=[np.float(label.split('_')[1]),np.float(label.split('_')[2]),np.float(label.split('_')[3])]
                label=euler.euler2quat(label[2]*np.pi/180.0,label[1]*np.pi/180.0,label[0]*np.pi/180.0)
                data.append([file,label])
        print("Fichiers detectes et etiquetes.")
        return data
    
    def getDataFromFiles(files,size,typeOfSet):
        print("Chargement des images...")
        data=[]
        av=5
        
        f=open(PATH+'/'+typeOfSet+'/'+'groundTruth.data','rb')
        groundTruth=pickle.load(f)
        f.close()
        
        for i,file in enumerate(files):
            if depth==1:
                for gT in groundTruth:
                    if gT[0]==file[0]:
                        img=np.zeros([2,size,size])
                        
                        imgT=cv2.imread(PATH+'/'+typeOfSet+'/'+file[0],0)
                        imgT=crop(imgT,gT[1:5])
                        imgT=cv2.resize(imgT,(size,size))
                        img[0,:,:]=imgT
                        
                        
                        '''
                        # Plot the original and the radon transformed image
                        plt.subplot(1, 2, 1), plt.imshow(imgT, cmap='gray')
                        plt.xticks([]), plt.yticks([])
                        plt.subplot(1, 2, 2), plt.imshow(radonT, cmap='gray')
                        plt.xticks([]), plt.yticks([])
                        plt.show()
                        '''
                        
                        imgT=cv2.imread(PATH+'/'+typeOfSet+'/'+file[0].split('.png')[0]+'.exr',0)*255
                        imgT=crop(imgT,gT[1:5])
                        imgT=cv2.resize(imgT,(size,size))
                        radonT = radon.discrete_radon_transform(imgT, size)
                        img[1,:,:]=radonT
                        
                        #def crop(img,groundTruth,delta=5):
                        #imgT=cv2.resize(imgT,(size,size))
                        #img[2,:,:]=imgT
                        ##TEST AVEC RADON ?
                        

                        #cv2.imshow('test',np.matrix(img[0,:,:],np.uint8))
                        #cv2.waitKey()
            else:
                img=np.zeros([1,size,size])
                imgT=cv2.imread(PATH+'/'+typeOfSet+'/'+file[0],0)
                imgT=cv2.resize(imgT,(size,size))
                img[0,:,:]=imgT
            data.append([img,file[1]])
            if int(100.0*i/len(files))>=av:
                print(str(int(100.0*i/len(files))) + '%')
                av+=5
        print("Images chargees.")
        return data
    
    def exportData(data,typeOfSet):
        print("Creation du dictionnaire...")
        dictionnary=dict()
        dictionnary['data']=[]
        dictionnary['label']=[]
        for d in data:
            
            dictionnary['data'].append(d[0])
            quat1=d[1][0]
            quat2=d[1][1]
            quat3=d[1][2]
            quat4=d[1][3]
            print(quat1)
            dictionnary['label'].append([quat1,quat2,quat3,quat4])
        pickle.dump(dictionnary, open(PATH+'/'+typeOfSet+'.data', "wb" ) )
        print("Dictionnaire cree.")
    
    if training==1:
        files=getFiles('TRAIN')
        data=getDataFromFiles(files,size,'TRAIN')
        exportData(data,'TRAIN')
    if validation==1:
        files=getFiles('VALIDATION')
        data=getDataFromFiles(files,size,'VALIDATION')
        exportData(data,'VALIDATION')    
    if test==1:
        files=getFiles('TEST')
        data=getDataFromFiles(files,size,'TEST')
        exportData(data,'TEST')   