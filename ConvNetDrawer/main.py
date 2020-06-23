from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense
from pptx_util import save_model_to_pptx
from matplotlib_util import save_model_to_file

model = Model(input_shape=(1000,40,1))

model.add(Conv2D(32, (7, 7), (1, 1),padding= "valid")) # stride 1 1 por defecto
# model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding="same"))
model.add(Conv2D(64, (5, 5), padding="same"))
# model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), (2,2) ,padding="same"))
# model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3), strides=(2, 2),padding="same"))
model.add(Conv2D(256, (3, 3), padding="same"))
# model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3), strides=(2, 2),padding="same"))
model.add(Conv2D(512, (3, 3), padding="same"))
# model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3), strides=(2, 2),padding="same"))
model.add(Flatten())
# model.add(BatchNormalization())
model.add(Dense(256))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
model.add(Dense(3))





# model.add(Conv2D(384, (3, 3), padding="same"))
# model.add(Conv2D(384, (3, 3), padding="same"))
# model.add(Conv2D(256, (3, 3), padding="same"))
# model.add(MaxPooling2D((3, 3), strides=(2, 2)))
# model.add(Flatten())
# model.add(Dense(4096))
# model.add(Dense(4096))
# model.add(Dense(1000))


# model.add(Conv2D(64,(5,5), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
# model.add(Conv2D(128,(3,3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
# model.add(Conv2D(256,(3,3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
# model.add(Conv2D(512,(3,3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
# model.add(Flatten())
# model.add(BatchNormalization())
# model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(3, activation='softmax'))


# save as svg file
model.save_fig("example.svg")

# save as pptx file
save_model_to_pptx(model, "example.pptx")

# save via matplotlib
save_model_to_file(model, "example.pdf")