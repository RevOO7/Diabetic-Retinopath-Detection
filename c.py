import numpy as np
from keras.models import load_model
import cv2
from keras.optimizers import SGD,RMSprop
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator

# #Input Image
# target_directory = 'uploader/2.png'
IMG_DIM = 256
#Load Image
def chest(img_path):        
        #Pre-processing
        def crop_image_from_gray(img,tol=7):
            if img.ndim ==2:
                mask = img>tol
                return img[np.ix_(mask.any(1),mask.any(0))]
            elif img.ndim ==3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                mask = gray_img>tol
        
            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0): # image is too dark so that we crop out everything,
                return img # return original image
            else:
                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
                img = np.stack([img1,img2,img3],axis=-1)
            return img
        
        def circle_crop(img):   
     
            height, width, depth = img.shape
            largest_side = np.max((height, width))
            img = cv2.resize(img, (largest_side, largest_side))  
    
            x = int(width/2)
            y = int(height/2)
            r = np.amin((x,y))
    
            circle_img = np.zeros((height, width), np.uint8)
            cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
            img = cv2.bitwise_and(img, img, mask=circle_img)
    
            return img 
        
        def load_ben_color(image, sigmaX=10):
            #image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = cv2.resize(image, (IMG_DIM, IMG_DIM))
            image = crop_image_from_gray(image)
            image = cv2.resize(image, (IMG_DIM, IMG_DIM))
            image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
            image = circle_crop(image)  
            return image
        
        #classifier.summary()

        #Single Prediction        
        test_img = cv2.imread(img_path)
        test_img = cv2.resize(test_img,(IMG_DIM,IMG_DIM))
        test_img = np.expand_dims(test_img, axis=0) 
        dummy_datagen=ImageDataGenerator(rescale=1./255, preprocessing_function=load_ben_color)
        dummy_generator = dummy_datagen.flow(test_img, y=None, batch_size=1, seed=7)
        
        preds = classifier.predict_generator(generator=dummy_generator, steps=1)
        predicted_class_indices = np.argmax(preds, axis=1)

        #Label Dictionary
        label_maps = {0: 'No DR', 1: 'Non-Proliferative DR', 2: 'Proliferative DR'}
        label = label_maps[int(predicted_class_indices)]
        
        return(str(preds))