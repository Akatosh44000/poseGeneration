'''
@author: j.langlois
'''

import numpy as np
import bpy
import bpy_extras
import os, sys
import pickle
import imageio
import skimage.transform
import math

DATASET_PATH='/home/akatosh/DATASETS'
MODEL_PATH='MULTITUDE/BREATHER'

training=int(sys.argv[5])
validation=int(sys.argv[6])
test=int(sys.argv[7])
rX=int(sys.argv[8])
rY=int(sys.argv[9])
rZ=int(sys.argv[10])

sys.stdout = sys.stderr

def progressionBar(progress,total,current,symbol='|'):
    if int(100*current/total)>(progress+4):
        progress=int(100*current/total)
    if progress==100:
        sys.stdout.write("  "+str(progress)+"% "+symbol*(int(progress/5))+"\n")
        return progress
    else:
        sys.stdout.write("  "+str(progress)+"% "+symbol*(int(progress/5))+"\r")
        sys.stdout.flush()
    return progress

def euler2quat(z=0, y=0, x=0):
    [z,y,x]=[z/2.0,y/2.0,x/2.0]
    [cz,cy,cx] = [math.cos(z),math.cos(y),math.cos(x)]
    [sz,sy,sx] = [math.sin(z),math.sin(y),math.sin(x)]
    return np.array([cx*cy*cz - sx*sy*sz,cx*sy*sz + cy*cz*sx,cx*cz*sy - sx*cy*sz,cx*cy*sz + sx*cz*sy])
    
class InstanceGenerator():
    def __init__(self,axes=[1,1,1],training=True,validation=True,test=True):
        print("## INITATING GENERATOR...")
        bpy.ops.import_mesh.stl(filepath="/home/akatosh/DATASETS/MULTITUDE/BREATHER/breather.STL")
        self.ob = bpy.data.objects['breather']
        self.scene = bpy.context.scene
        bpy.context.scene.render.engine = 'BLENDER_RENDER'
        mat = bpy.data.materials.new('TexBreather')
        mat.diffuse_color=[0,0,0]
        me = self.ob.data
        me.materials.append(mat)
        self.cam=bpy.data.objects['Camera']
        [self.rX,self.rY,self.rZ]=axes
        self.training=training
        self.validation=validation
        self.test=test
        self.PATH=DATASET_PATH+'/'+MODEL_PATH+'/SAMPLES'
        self.maxTHETA=180
        self.maxPHI=180
        self.maxPSI=360
        self.size=64
        print("--- MAX THETA : "+str(self.maxTHETA))
        print("--- MAX PHI : "+str(self.maxPHI))
        print("--- MAX PSI : "+str(self.maxPSI))
        print("## GENERATOR INITIATED.\n")
        
    def cleanFolders(self):
        print("## CLEANING FOLDERS...")
        def clean(folder_name):
            for the_file in os.listdir(folder_name):
                file_path = os.path.join(folder_name, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)     
        clean('DEPTH')
        clean('STANDARD')
        if self.training==1:
            clean(self.PATH+'/'+'TRAIN')
        if self.validation==1:
            clean(self.PATH+'/'+'VALIDATION')
        if self.test==1:
            clean(self.PATH+'/'+'TEST')
        print("## FOLDERS CLEANED.\n")
        
    def generateInstance(self,dataset,iteration,angles=[0,0,0],light="default"):
        self.ob.rotation_euler=[angles[0]*np.pi/180.0,angles[1]*np.pi/180.0,angles[2]*np.pi/180.0]
        self.ob.location=[0,0,(np.random.random()-0.5)*0]
        bpy.data.lamps['Lamp'].type='HEMI' 
        
        bpy.ops.render.render(use_viewport=False)
        verts = ((self.ob.matrix_world * v.co) for v in self.ob.data.vertices)
        coords_2d = [bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, coord) for coord in verts]
        coords_2d= np.matrix(coords_2d)[:,0:2]  
        render_scale = self.scene.render.resolution_percentage / 100
        render_size = (int(self.scene.render.resolution_x * render_scale),int(self.scene.render.resolution_y * render_scale))
        coords_2dF=coords_2d
        coords_2dF[:,0]=coords_2d[:,0] * render_size[0]
        coords_2dF[:,1]=coords_2d[:,1] * render_size[1]
        boundingBox=[int(np.max(coords_2dF[:,0])),int(np.min(coords_2dF[:,0])),
                     int(np.max(coords_2dF[:,1])),int(np.min(coords_2dF[:,1]))]
        os.rename('STANDARD/Image0001.png', self.PATH+'/'+dataset+'/'+str(int(iteration))+'.png')
        os.rename('DEPTH/Image0001.exr', self.PATH+'/'+dataset+'/'+str(int(iteration))+'.exr')
        return boundingBox
    
    def iterateAngles(self,dataset,step,groundTruth):
        print("--- ANGLES INTERVAL : "+str(self.rX*step[0])+","+str(self.rY*step[1])+","+str(self.rZ*step[2]))
        incTHETA=step[0]
        incPHI=step[1]
        incPSI=step[2]
        incX=self.rX*(int(self.maxTHETA/incTHETA)+1)+np.abs(rX-1)
        incY=self.rY*(int(self.maxPHI/incPHI)+1)+np.abs(rY-1)
        incZ=self.rZ*(int(self.maxPSI/incPSI)+1)+np.abs(rZ-1)
        nbImages=(incX)*(incY)*(incZ)
        print("--- GENERATING "+str(nbImages)+" IMAGES...")
        nbImagesDone=0
        progress=0
        for angleX in range(0,incX):
            for angleY in range(0,incY):
                for angleZ in range(0,incZ):
                    #LIGHTING VARIATION ?
                    angles=[angleX*incTHETA,angleY*incPHI,angleZ*incPSI]       
                    boundingBox=self.generateInstance(dataset,nbImagesDone,angles)
                    self.groundTruthCompletion(nbImagesDone,angles,boundingBox,groundTruth)
                    nbImagesDone+=1
                    progress=progressionBar(progress,nbImages,nbImagesDone)
        print("--- "+str(nbImages)+" IMAGES GENERATED.")                
                        
    def randomAngles(self,dataset,nbTest,groundTruth):
        print("--- NUMBER OF EXAMPLES : "+str(nbTest))
        print("--- GENERATING "+str(nbTest)+" IMAGES...")
        progress=0
        for i in range(nbTest):
            angleX=self.rX*np.random.random()*self.maxTHETA
            angleY=self.rY*np.random.random()*self.maxPHI
            angleZ=self.rZ*np.random.random()*self.maxPSI
            angles=[angleX,angleY,angleZ]
            boundingBox=self.generateInstance(dataset,i,angles)
            self.groundTruthCompletion(i,angles,boundingBox,groundTruth)
            progress=progressionBar(progress,nbTest,i)
        print("--- "+str(nbTest)+" IMAGES GENERATED.")  
        
    def groundTruthCompletion(self,iteration,angles,boundingBox,groundTruth):
        groundTruth.append([int(iteration),boundingBox[0],boundingBox[1],boundingBox[2],boundingBox[3],angles[0],angles[1],angles[2]])
        
    def cropImage(self,dataset,groundTruth,index,bbSize):
        image=imageio.imread(self.PATH+'/'+dataset+'/'+str(int(groundTruth[index,0]))+'.png')
        imageD=imageio.imread(self.PATH+'/'+dataset+'/'+str(int(groundTruth[index,0]))+'.exr')
        boudingBox=[image.shape[0]-int(groundTruth[index,3]),image.shape[0]-int(groundTruth[index,4]),
                    int(groundTruth[index,2]),int(groundTruth[index,1])]
        cX=int(np.sum(boudingBox[0:2])/2)
        cY=int(np.sum(boudingBox[2:4])/2)
        image=image[int(cX-bbSize/2):int(cX+bbSize/2),int(cY-bbSize/2):int(cY+bbSize/2),:]
        image=skimage.transform.resize(image,(self.size,self.size), mode='reflect',preserve_range=True)
        imageD=imageD[int(cX-bbSize/2):int(cX+bbSize/2),int(cY-bbSize/2):int(cY+bbSize/2),:]
        imageD=skimage.transform.resize(imageD,(self.size,self.size), mode='reflect',preserve_range=True)
        #SKIMAGE RETURNS A FLOAT64 VERSION OF THE INPUT AND WE CAN SAVE IN FLOAT32 MAX
        imageUINT8=np.empty(image.shape,np.uint8)
        imageDF32=np.empty(image.shape,np.float32)
        for j in range(4):
            imageUINT8[:,:,j]=np.matrix(image[:,:,j],np.uint8)
            imageDF32[:,:,j]=np.matrix(imageD[:,:,j],np.float32)
        imageio.imwrite(self.PATH+'/'+dataset+'/'+str(int(groundTruth[index,0]))+'.png', imageUINT8)
        imageio.imwrite(self.PATH+'/'+dataset+'/'+str(int(groundTruth[index,0]))+'.exr', imageDF32)
        
    def cropImages(self,dataset,delta=5):
        #CROP THE IMAGES WITH RESPECT TO A BOUNDING BOX BUT CAN LEAD TO AN UPSCALE
        f=open(self.PATH+'/'+dataset+'/groundTruth.data','rb')
        groundTruth=np.matrix(pickle.load(f))
        f.close()
        print("--- CROPPING "+str(groundTruth.shape[0])+" IMAGES (OWN BOUDING BOX)...")
        groundTruthF=np.matrix(groundTruth[:,1:5],np.int)
        progress=0
        for i in range(groundTruth.shape[0]):
            dX=np.max(groundTruthF[i,2]-groundTruthF[i,3])
            dY=np.max(groundTruthF[i,0]-groundTruthF[i,1])
            d=np.maximum(dX,dY)+delta
            self.cropImage(dataset,groundTruth,i,d)
            progress=progressionBar(progress,groundTruth.shape[0]-1,i)
        print("--- "+str(groundTruth.shape[0])+" IMAGES CROPPED.")
    
    def cropImagesWithMaxSize(self,dataset,delta=5):
        #CROP THE IMAGES WITH THE LARGEST BOUNDING BOX TO AVOID UPSCALING
        f=open(self.PATH+'/'+dataset+'/groundTruth.data','rb')
        groundTruth=np.matrix(pickle.load(f))
        f.close()
        print("--- CROPPING "+str(groundTruth.shape[0])+" IMAGES (LARGEST BOUDING BOX)...")
        groundTruthF=np.matrix(groundTruth[:,1:5],np.int)
        dX=np.max(groundTruthF[:,2]-groundTruthF[:,3])
        dY=np.max(groundTruthF[:,0]-groundTruthF[:,1])
        d=np.maximum(dX,dY)+delta
        progress=0
        for i in range(groundTruth.shape[0]):
            self.cropImage(dataset,groundTruth,i,d)
            progress=progressionBar(progress,groundTruth.shape[0]-1,i)
        print("--- "+str(groundTruth.shape[0])+" IMAGES CROPPED.")
    
    def createDictionnary(self,dataset):
        f=open(self.PATH+'/'+dataset+'/groundTruth.data','rb')
        groundTruth=np.matrix(pickle.load(f))
        f.close()
        print("--- EXPORTING "+str(groundTruth.shape[0])+" IMAGES WITH LABELS TO DICTIONNARY...")
        imageSet=[]
        labelSet=[]
        progress=0
        for i in range(groundTruth.shape[0]):
            instance=np.empty([2,self.size,self.size],np.float32)
            labelSet.append(euler2quat(groundTruth[i,4]*np.pi/180.0,
                                     groundTruth[i,5]*np.pi/180.0,
                                     groundTruth[i,6]*np.pi/180.0))
            image=imageio.imread(self.PATH+'/'+dataset+'/'+str(int(groundTruth[i,0]))+'.png')
            instance[0,:,:]=image[:,:,0]
            imageD=imageio.imread(self.PATH+'/'+dataset+'/'+str(int(groundTruth[i,0]))+'.exr')
            instance[1,:,:]=imageD[:,:,3]
            imageSet.append(instance)
            progress=progressionBar(progress,groundTruth.shape[0]-1,i)
        dictionnarySet=dict()
        dictionnarySet['data']=imageSet
        dictionnarySet['labels']=labelSet
        f=open(self.PATH+'/'+dataset+'.data','wb')
        pickle.dump(dictionnarySet,f)
        f.close()
        print("--- "+str(groundTruth.shape[0])+" IMAGES AND LABELS EXPORTED.")
                  
    def generateInstances(self,dataset):
        stepTr=[5,5,5]
        stepVa=[7,7,7]
        nbTest=500
        groundTruth=[]
        if dataset=='TRAIN':
            self.iterateAngles(dataset,stepTr,groundTruth)
        elif dataset=='VALIDATION':
            self.iterateAngles(dataset,stepVa,groundTruth)
        elif dataset=='TEST':
            self.randomAngles(dataset,nbTest,groundTruth)
        f=open(self.PATH+'/'+dataset+'/groundTruth.data','wb')
        pickle.dump(groundTruth,f)
        f.close()
        self.cropImagesWithMaxSize(dataset)
        self.createDictionnary(dataset)
            
    def generate(self):
        if self.training==1:
            print("## GENERATING TRAINING DATASET...")
            self.generateInstances('TRAIN')
            print("## TRAINING DATASET GENERATED.\n")
        if self.validation==1:
            print("## GENERATING VALIDATION DATASET...")
            self.generateInstances('VALIDATION')
            print("## VALIDATION DATASET GENERATED.\n")
        if self.test==1:
            print("## GENERATING TEST DATASET...")
            self.generateInstances('TEST')
            print("## TEST DATASET GENERATED.\n")

#START OF GENERATION
generator=InstanceGenerator([rX,rY,rZ],training,validation,test)
generator.cleanFolders()
generator.generate()
print("## END OF GENERATION.")
#END OF GENERATION
bpy.ops.wm.quit_blender()


