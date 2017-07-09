import datetime
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":
    # Convolutional Neural Network

    # Part 1 - Data Pre-processing
    # for our dogs/cats dataset, this work has already manually been done.

    # Part 2 - Building the CNN

    # Initialize our CNN
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

    # Step 2 - Max Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Add second convolutional layer to improve accuracy
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    # Step 5 - Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Part 3 - Fitting the CNN to the images

    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
    )

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('cnn_files/dataset/training_set',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('cnn_files/dataset/test_set',
                                                target_size = (64, 64),
                                                class_mode = 'binary')

    # get the initial timestamp before attempting the ANN
    time_0 = datetime.datetime.now()

    classifier.fit_generator(training_set,
                             steps_per_epoch = 8000,
                             epochs = 25,
                             validation_data = test_set,
                             validation_steps = 2000,
                             workers = 8)


    # grab the difference in time it took to run
    time_taken = datetime.datetime.now() - time_0

    print('###########################################################')
    print('###########################################################')
    print('###########################################################')
    print('')
    print('')
    print('')
    print('Time: ' + str(time_taken))
    print('')
    print('')
    print('')
    print('###########################################################')
    print('###########################################################')
    print('###########################################################')
