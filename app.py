import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
def has(arr,item):
    for i in arr:
        if (i == item):
            return True
    return False

listDir = os.listdir('.')
CHECKPOINTNAME = 'model.ckpt'
WIDTH = 256
HEIGHT = 256
mask = glob('./dataset/kaggle_3m/*/*_mask*')
train = []
'''
Data preProcessing 
'''
for i in mask:
    train.append(i.replace('_mask',''))
df = pd.DataFrame(data={
    'filename': train,
    'mask': mask
})
df_train,df_test = train_test_split(df, test_size=.1)
df_train,df_val = train_test_split(df_train,test_size=.2)

'''
U-net Generator
'''

def UNET(input_size=(256,256,3)):
    from tensorflow.keras import Input,Model
    from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Conv2DTranspose,concatenate
    inputs = Input(input_size)

    conv1 = Conv2D(64,(3,3), padding='same')(inputs)
    a1 = Activation('relu')(conv1)
    conv1 = Conv2D(64,(3,3), padding='same')(a1)
    bn = BatchNormalization(3)(conv1)
    bn = Activation('relu')(bn)
    pool1 = MaxPooling2D()(bn)

    conv2 = Conv2D(128,(3,3),padding='same')(pool1)
    bn2 = Activation('relu')(conv2)
    conv2 = Conv2D(128,(3,3),padding='same')(bn2)
    bn2 = BatchNormalization(3)(conv2)
    bn2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D()(bn2)
    

    conv3 = Conv2D(128,(3,3),padding='same')(pool2)
    bn3 = Activation('relu')(conv3)
    conv3 = Conv2D(128,(3,3),padding='same')(bn3)
    bn3 = BatchNormalization(3)(conv3)
    bn3 = Activation('relu')(bn3)
    pool3 = MaxPooling2D()(bn3)

    conv4 = Conv2D(128,(3,3),padding='same')(pool3)
    bn4 = Activation('relu')(conv4)
    conv4 = Conv2D(128,(3,3),padding='same')(bn4)
    bn4 = BatchNormalization(3)(conv4)
    bn4 = Activation('relu')(bn4)
    pool4 = MaxPooling2D()(bn4)

    conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
    bn5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, (3, 3), padding='same')(bn5)
    bn5 = BatchNormalization(axis=3)(conv5)
    bn5 = Activation('relu')(bn5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), padding='same')(up6)
    bn6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, (3, 3), padding='same')(bn6)
    bn6 = BatchNormalization(axis=3)(conv6)
    bn6 = Activation('relu')(bn6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), padding='same')(up7)
    bn7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, (3, 3), padding='same')(bn7)
    bn7 = BatchNormalization(axis=3)(conv7)
    bn7 = Activation('relu')(bn7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), padding='same')(up8)
    bn8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, (3, 3), padding='same')(bn8)
    bn8 = BatchNormalization(axis=3)(conv8)
    bn8 = Activation('relu')(bn8)

    up9 = concatenate([Conv2DTranspose(64,(2,2), strides=(2,2), padding='same')(bn8), conv1], axis=3)
    conv9 = Conv2D(64, (3,3), padding='same')(up9)
    bn9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, (3,3), padding='same')(bn9)
    bn9 = BatchNormalization(axis=3)(conv9)
    bn9 = Activation('relu')(bn9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn9)

    return Model(inputs=[inputs], outputs=[conv10])

def dataGenerator(dataFrame,batchSize,augDict,imageColorMode='rgb',
                        maskColorMode='grayscale',imgSavePrefix='image',maskSavePrefix='mask',
                        saveToDir=None,targetSize=(256,256),seed=1
                        ):
    imgDataGen = ImageDataGenerator(**augDict)
    maskDataGen = ImageDataGenerator(**augDict)
    imgGen = imgDataGen.flow_from_dataframe(
        dataFrame,
        x_col='filename',
        class_mode=None,
        batch_size=batchSize,
        color_mode=imageColorMode,
        target_size=targetSize,
        save_to_dir=saveToDir,
        save_prefix=imgSavePrefix,
        seed=seed
    )
    maskGen = maskDataGen.flow_from_dataframe(
        dataFrame,
        x_col='mask',
        class_mode=None,
        color_mode=maskColorMode,
        target_size=targetSize,
        batch_size=batchSize,
        save_to_dir=saveToDir,
        save_prefix=maskSavePrefix,
        seed=seed,
    )
    trainGen = zip(imgGen, maskGen)
    for (img,mask) in trainGen:
        img = img/ 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        yield(img,mask)

def trainModel():
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import ModelCheckpoint
    SMOOTH=1
    BATCHSIZE=32
    EPOCHS=150

    def dice_coef(y_true, y_pred):
        y_truef=K.flatten(y_true)
        y_predf=K.flatten(y_pred)
        And=K.sum(y_truef* y_predf)
        return((2* And + SMOOTH) / (K.sum(y_truef) + K.sum(y_predf) + SMOOTH))
        
    #Dice Loss
    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)
    def iou(y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        sum_ = K.sum(y_true + y_pred)
        jac = (intersection + SMOOTH) / (sum_ - intersection + SMOOTH)
        return jac

    tarinGeneratorArgs = dict()
    trainGen = dataGenerator(df_train, BATCHSIZE, tarinGeneratorArgs,targetSize=(WIDTH,HEIGHT))
    testGen = dataGenerator(df_val, BATCHSIZE,
                                dict(),
                                targetSize=(WIDTH, HEIGHT))
    # model = keras.models.load_model('./Unet.h5',custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef, 'iou': iou})
    # print(model.summary())
    # res = model.evaluate(testGen, steps=88)
    model = UNET(input_size=(WIDTH,HEIGHT,3))
    checkpoint = ModelCheckpoint(
                                    filepath='./model.ckpt',
                                    monitor='val_acc', 
                                    save_weights_only=False, 
                                    save_best_only=True, 
                                    mode='auto',
                                    period=1
                                )
    model.compile(
        optimizer='adam',
        loss=dice_coef_loss,
        metrics=["binary_accuracy", iou,dice_coef]
    )
    history = model.fit(trainGen,
                    steps_per_epoch=len(df_train) / BATCHSIZE, 
                    epochs=EPOCHS, 
                    validation_data = testGen,
                    validation_steps=len(df_val) / BATCHSIZE,
                    callbacks=[checkpoint])
trainModel()
