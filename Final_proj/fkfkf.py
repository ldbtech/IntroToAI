model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=32, input_shape=(28, 28, 1), kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))

# Max Pooling
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

# Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(512))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.3))

# Output Layer
# important to have dense 10, since we have 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])


#################################################
# Step 4:
# Prepare for training

# We use ImageDataGenerator to augment our input data
# which among other benefits, can help reduce over-fitting
gen = ImageDataGenerator(
            rotation_range=8,
            width_shift_range=0.08,
            shear_range=0.3,
            height_shift_range=0.08,
            zoom_range=0.08
            )
test_gen = ImageDataGenerator()

# hyoer-parameters
# We train in batches to speed up the process
# (and so that our memory can handle the data)
BATCH_SIZE = 64
# How many rounds of training? Let's start from a smaller number
EPOCHS = 5

# Generator to "flow" in the input images and labels into our model
# Takes batch_size as a parameter
train_generator = gen.flow(X_train, y_train, batch_size=BATCH_SIZE)
test_generator = test_gen.flow(X_test, y_test, batch_size=BATCH_SIZE)

#################################################
# Step 5:
# Do the training!
model.fit_generator(
        train_generator,
        steps_per_epoch=60000//BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_generator,
        validation_steps=10000//BATCH_SIZE
        )