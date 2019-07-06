import keras
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Activation, Convolution2D, Permute, Reshape, multiply, add, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt

c = 23
# se-block
def squeeze_excite_block(input, ratio=16):
    init = input
    filters = int(init.shape[-1])
    print(type(filters))
    se_shape = (1, 1, filters)
    res = int(filters // ratio)
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(res, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)


    x = multiply([init, se])
    return x

# input
input_data = Input(shape=[200, 200, 3])
# first layer
conv1 = Convolution2D(filters=32, kernel_size=[5, 5])(input_data)
BN1 = BatchNormalization()(conv1)
se1 = squeeze_excite_block(BN1)
ac1 = Activation('relu')(se1)
pool1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(ac1)

# second layer
conv2 = Convolution2D(filters=64, kernel_size=[3, 3])(pool1)
BN2 = BatchNormalization()(conv2)
se2 = squeeze_excite_block(BN2)
ac2 = Activation('relu')(se2)
pool2 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(ac2)

# third layer
conv3 = Convolution2D(filters=64, kernel_size=[3, 3])(pool2)
BN3 = BatchNormalization()(conv3)
se3 = squeeze_excite_block(BN3)
ac3 = Activation('relu')(se3)
pool3 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(ac3)

# forth layer
conv4 = Convolution2D(filters=128, kernel_size=[2, 2])(pool3)
BN4 = BatchNormalization()(conv4)
se4 = squeeze_excite_block(BN4)
ac4 = Activation('relu')(se4)
pool4 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(ac4)

# fifth layer
conv5 = Convolution2D(filters=256, kernel_size=[2, 2])(pool4)
BN5 = BatchNormalization()(conv5)
se5 = squeeze_excite_block(BN5)
ac5 = Activation('relu')(se5)
pool5 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(ac5)

# flatten
flatten = Flatten()(pool5)

# fc1
fc1 = Dense(units=256)(flatten)
fc_bn1 = BatchNormalization()(fc1)
fc_ac1 = Activation('relu')(fc_bn1)

# fc2
fc2 = Dense(units=128)(fc_ac1)
fc_bn2 = BatchNormalization()(fc2)
fc_ac2 = Activation('relu')(fc_bn2)
dp = Dropout(0.5)(fc_ac2)

# fc3
fc3 = Dense(units=c)(dp)
fc_bn3 = BatchNormalization()(fc3)
output = Activation('softmax')(fc_bn3)

model = Model(inputs=input_data, outputs=output)

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# prepare data
train_dir = 'fish_image'  # 训练集数据


nb_epoches = int(50)                # epoch数量
batch_size = int(16)





#　图片生成器
train_datagen = ImageDataGenerator(rescale=1./255)
#validation_split=0.3 可以通过设置validation_split只从train文件中提取训练集和验证集（1）
valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)


# In[7]:


# 训练数据与测试数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')#subset='training'，假如设置了（1），可以通过设置subset来分配训练集和验证集

valid_generator = valid_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')#subset='validation'

# checkpoint
filepath = "CNN_SENet.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
if os.path.exists(filepath):
    model.load_weights(filepath)
    # 若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")


history_tl = model.fit_generator(
train_generator,
nb_epoch=nb_epoches,
validation_data=valid_generator,
steps_per_epoch= STEP_SIZE_TRAIN,
validation_steps = STEP_SIZE_VALID,
class_weight='auto',
callbacks=callbacks_list)

# draw result
fig, ax = plt.subplots(2,1,figsize=(10,10))
ax[0].plot(history_tl.history['loss'], color='r', label='Training Loss')
ax[0].plot(history_tl.history['val_loss'], color='g', label='Validation Loss')
ax[0].legend(loc='best',shadow=True)
ax[0].grid(True)

ax[1].plot(history_tl.history['acc'], color='r', label='Training Accuracy')
ax[1].plot(history_tl.history['val_acc'], color='g', label='Validation Accuracy')
ax[1].legend(loc='best',shadow=True)
ax[1].grid(True)
plt.savefig("result+"+str(nb_epoches)+".png")
plt.show()
