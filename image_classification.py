# -*- coding: utf-8 -*-
'''
@title:          image_classification.py

@description:    This python script implements image classification algorithm.
                 It is based on Convolutional Neural Network Architecture and
                 uses PCA for dimensionality reduction of preprocessed images.
                 
@algorithm:      1. Identify and remove corrupt images.
                 2. Preprocess all train images.
                        a. Read image from disk.
                        b. Transform image colorspace, if specified.
                           Note: For the best performance, keep 'RGB' colorspace.
                        c. Normalize image data.
                        d. Transform numpy array elements into numpy.float32
                        e. Resize image to specified standard resolution.
                           Note: For experimentation purpose, 
                                   we have kept it 64x64 only.
                        f. Flatten numpy array and add it to memmap.
                 3. Apply PCA for dimensionality reduction.
                    For experimentation purpose, 
                        we are generating (Channels*X*Y) components, 
                        i.e. 3*32*32 components.
                 4. Build a Convolutional Neural Network Architecture/Model
                    with specified hyperparameters and techniques.
                 5. Train a model with preprocessed training images.
                 6. Preprocess all test images as specified in (1).
                 7. Predict labels for preprocessed testing images 
                     using the trained neural network model.
                 8. Summarize and save predictions.
                     
@author:         Chetan Borse

@date:           05/04/2016

@usage:          1. Simple usage   => 
                         ipython image_classification.py
                 2. Advanced usage => 
                         ipython image_classification.py -- 
                                                         -i <trainpath>
                                                         -o <testpath> 
                                                         -x <traincsv> 
                                                         -y <testcsv> 
                                                         -m <trainexamples> 
                                                         -n <testexamples>
                                                         -p <printsummary>
                                                         
@python_version: 3.5
===============================================================================
'''

import os
import sys
import cv2
import getopt
import warnings
import numpy as np
import pylab as plt
from pylab import contour, axis
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.metrics import multiclass_logloss
from sklearn.decomposition import PCA
from sklearn import metrics

warnings.filterwarnings("ignore")


class CSVData:
    """
    CSV data extraction tool for reading, cleaning and transforming data.
    """
    
    def __init__(self,
                 file,
                 imageid=range(1),
                 labels=range(1, 9),
                 multiclass=False):
        """
        Initialise CSV data extraction tool.
        
        @param: file       => File path
        @param: imageid    => Image id
        @param: labels     => Labels
        @param: multiclass => Flag to pick multiclass labels
        """
        self.file       = os.path.abspath(file)
        self.imageid    = list(imageid)
        self.labels     = list(labels)
        self.multiclass = multiclass
        self.csvdata    = None
        self.XTrain     = None
        self.YTrain     = None
        self.XTest      = None
        self.YTest      = None
        
    def read(self):
        """
        Read CSV data.
        """
        self.csvdata = np.genfromtxt(self.file,
                                     dtype=np.str,
                                     delimiter=",")

    def split(self, trainExample=1.0, testExample=0.0, shuffle=False):
        """
        Split CSV data into Training and Testing.

        @param: trainExample => Total training examples 
        @param: testExample  => Total test examples
        @param: shuffle      => Shuffle records
        """
        if self.csvdata == None:
            return

        # Total training examples
        if (trainExample <= 1):
            trainExample = np.round(trainExample * len(self.csvdata))

        # Total test examples
        if (testExample <= 1):
            testExample = np.round(testExample * len(self.csvdata))

        # Shuffle records
        if shuffle == True:
            np.random.shuffle(self.csvdata)
        
        # Images for Training and Testing
        self.XTrain = self.csvdata[:trainExample, self.imageid]
        self.XTest  = self.csvdata[(len(self.csvdata) - testExample):, self.imageid]
        if len(self.imageid) == 1:
            self.XTrain = self.XTrain.flatten()
            self.XTest  = self.XTest.flatten()

        # Label Matrix for Training and Testing
        self.YTrain = self.csvdata[:trainExample, self.labels]
        self.YTest  = self.csvdata[(len(self.csvdata) - testExample):, self.labels]

        # Label Vector/Matrix for Training and Testing
        if not self.multiclass:
            if self.YTrain.shape[1] > 1:
                YTrain = []
                for i, label in enumerate(self.YTrain):
                    YTrain.append(np.where(self.YTrain[i, :] == '1')[0])
                self.YTrain = np.array(YTrain)
            self.YTrain = self.YTrain.flatten()

        if not self.multiclass:
            if self.YTest.shape[1] > 1:
                YTest = []
                for i, label in enumerate(self.YTest):
                    YTest.append(np.where(self.YTest[i, :] == '1')[0])
                self.YTest = np.array(YTest)
            self.YTest = self.YTest.flatten()

        # Fix data type
        self._fixDataType()

    def _fixDataType(self):
        """
        Fix data type.
        """
        self.YTrain = self.YTrain.astype(np.uint8)
        self.YTest  = self.YTest.astype(np.uint8)

    def GetTrainCluster(self, label=0, examples=50000):
        """
        Get cluster of training examples with specific label.
        """
        return np.where(self.YTrain[:examples] == label)[0]

    def GetTestCluster(self, label=0, examples=50000):
        """
        Get cluster of testing examples with specific label.
        """
        return np.where(self.YTest[:examples] == label)[0]


class ImageData:
    """
    Image data extraction tool for reading, preprocessing and 
    transforming image.
    """
    
    def __init__(self,
                 folder,
                 imageids=None,
                 standardResolution=(256, 256),
                 normalize=True,
                 colorspace="RGB",
                 sift=True,
                 keypoints=128,
                 contour=False,
                 applyPCA=True,
                 pca=None,
                 component=(32, 32),
                 imageMemmap="ImageMemmap.dat",
                 imagecount=None,
                 useCustomImageMap=False,
                 filterids=[],
                 maxcount=50000):
        """
        Initialise Image data extraction tool.
        
        @param: folder             => Image Folder
        @param: imageids           => Image ids
        @param: standardResolution => Standard Resolution
        @param: normalize          => Flag to normalize images
        @param: colorspace         => Flag to transform colorspace of images
                                        1. GRAYSCALE transformation, 
                                            if selected "GRAYSCALE".
                                        2. RGB transformation, 
                                            if selected "RGB".
        @param: sift               => Flag to generate SIFT features for images
                                      Note: Currently, this feature is broken.
        @param: keypoints          => Maximum keypoints to be generated 
                                        for SIFT features
                                      Note: Currently, this feature is broken.
        @param: contour            => Flag to find contour plot in images
                                      Note: Currently, this feature is broken.
        @param: applyPCA           => Flag to apply PCA to images
        @param: pca                => Learned PCA model
        @param: component          => PCA components
                                        i.e. Spatial size, (Rows, Columns)
        @param: imageMemmap        => Memmap file 
                                        where preprocessed image data is stored
                                        on disk
        @param: imagecount         => Total images
        @param: useCustomImageMap  => Flag to load images 
                                        from existing memmap file
        @param: filterids          => Filter for bad images
        @param: maxcount           => Maximum images to be loaded
        """
        self.folder             = os.path.abspath(folder)
        self.imageids           = imageids
        self.standardResolution = standardResolution
        self.normalize          = normalize
        self.colorspace         = colorspace
        self.sift               = sift
        self.keypoints          = keypoints
        self.contour            = contour
        self.applyPCA           = applyPCA
        self.pca                = pca
        self.component          = component
        self.imageMemmap        = imageMemmap
        self.imagecount         = imagecount
        self.useCustomImageMap  = useCustomImageMap
        self.filterids          = filterids
        self.maxcount           = maxcount
        self.images             = None

    def load(self, displayImage=False):
        """
        Load image data.
        """
        # Set standard width and height for all images
        if self.sift:
            standardWidth  = self.keypoints
            standardHeight = 128
        else:
            standardWidth  = self.standardResolution[0]
            standardHeight = self.standardResolution[1]

        # If 'applyPCA' is True, 
        # then set number of PCA components along spatial surface
        if self.applyPCA:
            componentX = self.component[0]
            componentY = self.component[1]

        # If 'colorspace' is RGB, set channels == 3
        # If 'colorspace' is GRAYSCALE, set channels == 1
        if self.colorspace == "RGB":
            channels = 3
        elif self.colorspace == "GRAYSCALE":
            channels = 1

        if self.imagecount != None:
            imagecount = self.imagecount

        # If 'maxcount' is None, then load 50k images only
        if self.maxcount == None:
            self.maxcount = 50000

        # Do not load images and exit from function, 
        # if image memmap file is not provided or image folder is not provided
        if not self.useCustomImageMap and \
            self.folder == None and \
            self.imageids == None:
            return

        # List all images to be loaded from folder, 
        # if image memmap file is not provided
        if not self.useCustomImageMap and self.imageids == None:
            imagecount = 0
            imageids   = []
            imagepaths = []
            for root, dirs, images in os.walk(self.folder, topdown=False):
                for image in images:
                    if image.endswith(".jpg"):
                        if int(image.split(".jpg")[0]) not in self.filterids:
                            imageids.append(image.split(".jpg")[0])
                            imagepaths.append(os.path.join(root, image))
                            
                            imagecount += 1
                            if imagecount == self.maxcount:
                                break
            self.imageids = np.array(imageids)
        elif not self.useCustomImageMap and self.imageids != None:
            imagecount = 0
            imagepaths = []
            for imageid in self.imageids:
                imagepath = os.path.join(self.folder, imageid + ".jpg")
                if os.path.exists(imagepath):
                    if int(imageid) not in self.filterids:
                        imagepaths.append(imagepath)

                        imagecount += 1
                        if imagecount == self.maxcount:
                            break

        # Set custom image memmap, if it is not provided
        if not self.useCustomImageMap:
            imagemap = np.memmap(self.imageMemmap,
                                 dtype="float32",
                                 mode="w+",
                                 shape=(len(imagepaths),
                                        channels*standardWidth*standardHeight,
                                        )
                                 )
            del imagemap

        # Load images from folder itself, if image memmap file is not provided
        if not self.useCustomImageMap:
            imagecount   = 0
            memmapcount  = 0
            images       = None
            sift         = cv2.xfeatures2d.SIFT_create(self.keypoints)
            for imagepath in imagepaths:
                # Read image from disk
                image = cv2.imread(imagepath)
                if displayImage:
                    plt.imshow(image)
                    plt.show()

                # Transform image colorspace
                if self.colorspace == "RGB":
                    pass
                elif self.colorspace == "GRAYSCALE":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Normalize image data
                if self.normalize:
                    image = cv2.normalize(image.astype('float'), 
                                          None, 
                                          0.0, 
                                          1.0, 
                                          cv2.NORM_MINMAX)

                # Find SIFT features
                if self.sift:
                    kp, image = sift.detectAndCompute(image, None)

                # Find the contour plot in image
                if self.contour:
                    contour(image, 
                            levels=[255], 
                            colors='black', 
                            origin='image')
                    axis('equal')

                # Transform numpy array elements into numpy.float32
                image = image.astype(np.float32)

                # Resize image to standard resolution
                image = cv2.resize(image, (standardWidth, standardHeight))

                # Flatten numpy array
                image = np.squeeze(image.ravel())

                if images == None:
                    images = np.array([image])
                else:
                    images = np.append(images, [image], axis=0)

                imagecount += 1

                # Store batch of images into memmap array
                if (imagecount % 100) == 0:
                    imagemap = np.memmap(self.imageMemmap,
                                         dtype="float32",
                                         mode="r+",
                                         shape=(len(imagepaths),
                                                channels*standardWidth*standardHeight,
                                                )
                                         )
                    imagemap[memmapcount:imagecount] = images
                    del imagemap

                    memmapcount = imagecount
                    images      = None

                    print("Loaded ", imagecount, " images..")

            # Store last batch of images into memmap array
            if images != None:
                imagemap = np.memmap(self.imageMemmap,
                                     dtype="float32",
                                     mode="r+",
                                     shape=(len(imagepaths),
                                            channels*standardWidth*standardHeight,
                                            )
                                     )
                imagemap[memmapcount:imagecount] = images
                del imagemap

                memmapcount = imagecount
                images      = None

                print("Loaded ", imagecount, " images..")

        # Apply PCA if mentioned,
        # otherwise load images without any dimensionality reduction
        imagemap = np.memmap(self.imageMemmap, 
                             dtype="float32", 
                             mode="r+", 
                             shape=(imagecount,
                                    channels*standardWidth*standardHeight,
                                    )
                             )
        if self.applyPCA:
            if self.pca == None:
                self.pca = PCA(n_components=channels*componentX*componentY)
                self.pca.fit(imagemap)
            images      = self.pca.transform(imagemap)
            self.images = images.reshape((images.shape[0],
                                          channels,
                                          componentX,
                                          componentY))
        else:
            self.images = imagemap.reshape((imagemap.shape[0],
                                            channels,
                                            standardWidth,
                                            standardHeight))
        del imagemap


class EarlyStopping(object):
    """
    Class for early stopping.
    
    If validation error does not reduce even after specified number of epochs
    i.e. 'wait', stop training a model and return parameters of learned model 
    that has the least validation error.
    """
    
    def __init__(self, wait=25):
        self.wait            = wait
        self.best_valid_loss = np.inf
        self.best_epoch      = 0
        self.best_weights    = None

    def __call__(self, nn, train_history):
        current_valid_loss = train_history[-1]['valid_loss']
        current_epoch      = train_history[-1]['epoch']
        
        # Save parameters of the best model learned,
        # i.e. model having less validation error than the previous one
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.best_epoch      = current_epoch
            self.best_weights    = nn.get_all_params_values()
        # Stop training a model,
        # if validation error does not improve further even after specified number of epochs
        elif current_epoch > (self.best_epoch + self.wait):
            print("Early stopping..")
            print("Best validion loss: ", self.best_valid_loss)
            print("At epoch: ", self.best_epoch)
            nn.load_params_from(self.best_weights)
            raise StopIteration()


class PickBestModel(object):
    """
    Class for picking the best model.
    
    If training reaches to specified number of maximum epochs, 
    i.e. 'max_epochs', stop training a model and return the best model 
    that has the least validation error.
    """
    def __init__(self, max_epochs=25):
        self.max_epochs       = max_epochs
        self.best_valid_loss = np.inf
        self.best_epoch      = 0
        self.best_weights    = None
    
    def __call__(self, nn, train_history):
        current_valid_loss = train_history[-1]['valid_loss']
        current_epoch      = train_history[-1]['epoch']

        # Save parameters of the best model learned,
        # i.e. model having less validation error than the previous one
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.best_epoch      = current_epoch
            self.best_weights    = nn.get_all_params_values()
        
        # If training reaches to specified number of maximum epochs, 
        # i.e. 'max_epochs',
        # stop training a model and return the best model 
        # that has the least validation error.
        if current_epoch == self.max_epochs:
            print("Picking the best model..")
            print("Best validion loss: ", self.best_valid_loss)
            print("At epoch: ", self.best_epoch)
            nn.load_params_from(self.best_weights)
            raise StopIteration()


def BuildCNNModel(inputspace=(3, 256, 256),
                  labels=8,
                  learningrate=0.001,
                  momentum=0.95,
                  regression=False,
                  wait=50,
                  epochs=500,
                  logging=1):
    """
    Build a Convolutional Neural Network Architecture/Model.
    
    1. Layer Arrangements:
        input -> dropout -> (conv2d -> maxpool2d)*2 -> conv2d -> dropout 
                                                            -> dense -> output

    2. Input Layer:
        Input shape = As specified by user, i.e. (Channels, Width, Height)

    3. Conv2DLayer:
        Number of filters = 32/64
        Filter size       = 5x5
        Stride            = 1x1
        Zero padding      = 1
        Weight sharing    = Enabled
        Nonlinearity function = Rectify activation function (ReLU)
        
    4. MaxPool2DLayer:
        Pool size    = 2x2
        Stride       = 2x2
        Zero padding = 0
        
    5. DropoutLayer:
        Dropout probability = 0.2/0.5

    6. DenseLayer:
        Activation units      = 512
        Nonlinearity function = Rectify activation function (ReLU)
        
    7. Output Layer:
        Activation units      = Total output classes as specified by user
        Nonlinearity function = Softmax activation function (softmax)
    
    8. Weight initialization/update method:
        Weight initialization = Xavier initialization 
                                (also, known as Glorot initialization) 
                                with weights sampled from the Uniform distribution.
                                Note: In lasagne package, this is the default method
                                    for 'Conv2DLayer' and 'DenseLayer'
        Weight update         = Nesterov momentum
        
    9. Other hyper parameters:
        Learning rate (=0.001) and Momentum (=0.95)
        
    10. Other techniques used:
        a. Early stopping
        b. Pick model with least validation error

    @param: inputspace   => Input shape for InputLayer
                                i.e. (Channels, Width, Height)
    @param: labels       => Number of output classes
    @param: learningrate => Learning Rate
    @param: momentum     => The amount of momentum to apply. 
                            Higher momentum results in smoothing 
                                over more update steps.
    @param: regression   => Flag for regression task
    @param: wait         => Number of epochs one should wait before early stopping
    @param: epochs       => Maximum epochs
    @param: logging      => Flag for logging Neural Network progress
    """
    model = NeuralNet(
                # Layer arrangements
                layers = [('input',            layers.InputLayer),
                          ('dropout_first',    layers.DropoutLayer),
                          ('conv2d_first',     layers.Conv2DLayer),
                          ('maxpool2d_first',  layers.MaxPool2DLayer),
                          ('conv2d_second',    layers.Conv2DLayer),
                          ('maxpool2d_second', layers.MaxPool2DLayer),
                          ('conv2d_third',     layers.Conv2DLayer),
                          ('dropout_second',   layers.DropoutLayer),
                          ('dense',            layers.DenseLayer),
                          ('output',           layers.DenseLayer),
                          ],
        
                # Input Layer
                input_shape = (None, 
                               inputspace[0], 
                               inputspace[1], 
                               inputspace[2]),

                # First Dropout Layer
                dropout_first_p = 0.2,

                # First 2D Convolutional Layer
                conv2d_first_num_filters  = 32,
                conv2d_first_filter_size  = (5, 5),
                conv2d_first_stride       = (1, 1),
                conv2d_first_pad          = 'same',
                conv2d_first_untie_biases = False,
                #conv2d_first_W            = lasagne.init.GlorotUniform(),
                conv2d_first_nonlinearity = lasagne.nonlinearities.rectify,

                # First 2D Max Pooling Layer
                maxpool2d_first_pool_size = (2, 2),
                maxpool2d_first_stride    = (2, 2),
                maxpool2d_first_pad       = (0, 0),

                # Second 2D Convolutional Layer
                conv2d_second_num_filters  = 32,
                conv2d_second_filter_size  = (5, 5),
                conv2d_second_stride       = (1, 1),
                conv2d_second_pad          = 'same',
                conv2d_second_untie_biases = False,
                #conv2d_second_W            = lasagne.init.GlorotUniform(),
                conv2d_second_nonlinearity = lasagne.nonlinearities.rectify,

                # Second 2D Max Pooling Layer
                maxpool2d_second_pool_size = (2, 2),
                maxpool2d_second_stride    = (2, 2),
                maxpool2d_second_pad       = (0, 0),

                # Third 2D Convolutional Layer
                conv2d_third_num_filters  = 64,
                conv2d_third_filter_size  = (5, 5),
                conv2d_third_stride       = (1, 1),
                conv2d_third_pad          = 'same',
                conv2d_third_untie_biases = False,
                #conv2d_third_W            = lasagne.init.GlorotUniform(),
                conv2d_third_nonlinearity = lasagne.nonlinearities.rectify,

                # Second Dropout Layer
                dropout_second_p = 0.5,

                # Fully Connected Layer
                dense_num_units    = 512,
                #dense_W            = lasagne.init.GlorotUniform(),
                dense_nonlinearity = lasagne.nonlinearities.rectify,

                # Output Layer
                output_num_units    = labels,
                #output_W            = lasagne.init.GlorotUniform(),
                output_nonlinearity = lasagne.nonlinearities.softmax,

                # Weight Update Method
                update               = nesterov_momentum,
                update_learning_rate = learningrate,
                update_momentum      = momentum,

                # Objective Function
                #objective_loss_function = multiclass_logloss,

                # Regularization
                #objective_l2 = 0.0001,
                
                # Flag for regression task
                regression = regression,

                # Functions to be called after every epoch
                on_epoch_finished = [
                                     EarlyStopping(wait=wait),
                                     PickBestModel(max_epochs=epochs)
                                    ],

                # Maximum epochs
                max_epochs = epochs,

                # Logging
                verbose = logging,
                )
                
    return model


def main(argv):
    """
    Main routine of the script.
    """
    # Read arguments
    try:
        opts, args = getopt.getopt(argv, 
                                   "i:o:x:y:m:n:p:h",
                                   ["trainpath=", 
                                    "testpath=", 
                                    "traincsv=", 
                                    "testcsv=", 
                                    "trainexamples=", 
                                    "testexamples=",
                                    "printsummary=",
                                    "help="
                                    ]
                                   )
    except getopt.GetoptError:
        print('ipython image_classification.py -- \
                                               -i <trainpath> \
                                               -o <testpath> \
                                               -x <traincsv> \
                                               -y <testcsv> \
                                               -m <trainexamples> \
                                               -n <testexamples> \
                                               -p <printsummary>')
        sys.exit(2)

    trainpath     = "/My Computer/Academic/MS/Spring 2016/ML/Assignments/Project/original_data/project/train/"
    testpath      = "/My Computer/Academic/MS/Spring 2016/ML/Assignments/Project/original_data/project/test/"
    traincsv      = "/My Computer/Academic/MS/Spring 2016/ML/Assignments/Project/original_data/project/train.csv"
    testcsv       = None
    trainexamples = None
    testexamples  = None
    printsummary  = False
    for opt, arg in opts:
        if opt == '-h':
            print('ipython image_classification.py -- \
                                                   -i <trainpath> \
                                                   -o <testpath> \
                                                   -x <traincsv> \
                                                   -y <testcsv> \
                                                   -m <trainexamples> \
                                                   -n <testexamples> \
                                                   -p <printsummary>')
            sys.exit()
        elif opt in ("-i", "--trainpath"):
            trainpath = arg
        elif opt in ("-o", "--testpath"):
            testpath = arg
        elif opt in ("-x", "--traincsv"):
            traincsv = arg
        elif opt in ("-y", "--testcsv"):
            testcsv = arg
        elif opt in ("-m", "--trainexamples"):
            trainexamples = int(arg)
        elif opt in ("-n", "--testexamples"):
            testexamples = int(arg)
        elif opt in ("-p", "--printsummary"):
            printsummary = bool(arg)
    
    # Read training data from csv and preprocess it
    print('Reading and preprocessing training data from csv..')
    traincsv = CSVData(traincsv, multiclass=False)
    traincsv.read()
    traincsv.split(trainExample=1.0, testExample=0.0, shuffle=False)
    if trainexamples == None:
        XTrain = traincsv.XTrain
        YTrain = traincsv.YTrain
    else:
        XTrain = traincsv.XTrain[:trainexamples]
        YTrain = traincsv.YTrain[:trainexamples]
    print('Reading and preprocessing training data from csv is finished!\n')

    # Read testing data from csv and preprocess it
    if testcsv != None:
        print('Reading and preprocessing testing data from csv..')
        testcsv = CSVData(testcsv)
        testcsv.read()
        testcsv.split(trainExample=0.0, testExample=1.0, shuffle=False)
        if testexamples == None:
            XTest = testcsv.XTest
            YTest = testcsv.YTest
        else:
            XTest = testcsv.XTest[:testexamples]
            YTest = testcsv.YTest[:testexamples]
        print('Reading and preprocessing testing data from csv is finished!\n')

    # Read training images and preprocess it
    print('Reading and preprocessing image data..')
    badTrain  = [8807, 10668, 11402, 13585, 15553, 36911, 48624]
    traindata = ImageData(trainpath, 
                          imageids=XTrain,
                          standardResolution=(64, 64),
                          colorspace="RGB",
                          contour=False,
                          sift=False,
                          applyPCA=True,
                          component=(32, 32),
                          imageMemmap="train.dat",
                          imagecount=None,
                          useCustomImageMap=False,
                          filterids=badTrain)
    traindata.load()
    print('Reading and preprocessing image data is finished!\n')

    # Build a neural network model
    print("Building a model..")
    cnnModel = BuildCNNModel(inputspace=(3, 32, 32),
                             labels=8,
                             learningrate=0.001,
                             momentum=0.95,
                             regression=False,
                             wait=28,
                             epochs=350,
                             logging=1)
    print("Building a model is finished!\n")

    # Train a neural network model
    print("Training a model..")
    model = cnnModel.fit(traindata.images, YTrain)
    print("Training a model is finished!\n")

    # Clean some memory
    del traindata.images

    # Read testing images and preprocess it
    print('Reading and preprocessing image data..')
    badTest  = [60968, 64800, 72751, 73726, 75380, 76787, 79616, 72363]
    testdata = ImageData(testpath,
                         #imageids=XTest,
                         standardResolution=(64, 64),
                         colorspace="RGB",
                         contour=False,
                         sift=False,
                         applyPCA=True,
                         pca=traindata.pca,
                         component=(32, 32),
                         imageMemmap="test.dat",
                         imagecount=None,
                         useCustomImageMap=False,
                         filterids=badTest,
                         maxcount=testexamples)
    testdata.load()
    print('Reading and preprocessing image data is finished!\n')

    # Predict labels using the trained neural network model
    print('Predicting..')
    YPredicted = model.predict(testdata.images)
    print('Predicting is finished!\n')

    # Summarize the fit of the model
    if printsummary:
        YPredicted_1d = np.argmax(YPredicted, axis=1)
        print("Correct Predictions: ", np.sum(np.equal(YTest, YPredicted_1d)))
        print("Total Examples:      ", YTest.shape[0])
        print("\nClassification Report:")
        print(metrics.classification_report(YTest, YPredicted_1d))
        print("Confusion Matrix:")
        print(metrics.confusion_matrix(YTest, YPredicted_1d))

    # Save prediction results
    XTest  = np.reshape(testdata.imageids, (testdata.imageids.shape[0], 1))
    result = np.concatenate((XTest.astype("str"), YPredicted.astype("str")), axis=1)
    np.savetxt("result.csv", result, fmt="%s", delimiter=',')


if __name__ == '__main__':
    main(sys.argv[1:])