# ********************************
# HOTEL IMAGE CLASSIFICATION  *
# ********************************


* Introduction
* File List
* Requirements
* Modifications
* Usage
* Algorithm
* Maintainer


I. INTRODUCTION
---------------

This python script implements image classification algorithm.

It is based on Convolutional Neural Network Architecture and
uses PCA for dimensionality reduction of preprocessed images.


II. FILE LIST
-------------

a. image_classification.py      Image classification script
b. requirements.txt             Python dependencies to be installed
c. ImageClassification.docx	Documentation for Image Classification script
				(docx version)
d. ImageClassification.pdf	Documentation for Image Classification script
				(pdf version)
e. train.csv			Sample CSV format for training csv
f. result.csv			Sample csv containing predictions
g. README


III. REQUIREMENTS
-----------------

This application requires the following modules:
1. Anaconda (https://www.continuum.io/downloads)

2. CMake (https://cmake.org) (Preferably Version 3.1.0 and above)

   Steps to install CMake using ppa:
	> sudo apt-get remove cmake cmake-data
	> sudo -E add-apt-repository -y ppa:george-edison55/cmake-3.x
	> sudo -E apt-get update
	> sudo apt-get install cmake

3. OpenCV (http://opencv.org/downloads.html)

   Steps to install OpenCV using source:
	> cmake -D CMAKE_BUILD_TYPE=RELEASE 
             -D CMAKE_INSTALL_PREFIX=/usr/local 
             -D PYTHON3_PACKAGES_PATH=/Users/cborse/anaconda/lib/python3.5/site-packages 
             -D PYTHON3_LIBRARY=/Users/cborse/anaconda/lib/python3.5/ 
             -D PYTHON3_INCLUDE_DIR=/Users/cborse/anaconda/include/python3.5m 
             -D INSTALL_C_EXAMPLES=ON 
             -D INSTALL_PYTHON_EXAMPLES=ON 
             -D BUILD_EXAMPLES=ON
             -D BUILD_opencv_python3=ON 
             -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules ..
	> make
	> make install

   Note: Above paths can change as per your Anaconda installation.

4. Install Theano, Lasagne, nolearn as below:

	> pip install -r requirements.txt

5. Mac OS or Linux (Not tested on Windows)


IV. MODIFICATIONS
-----------------

Below modifications are required in source code of softwares to be installed ->

1. OpenCV:
    Comment below lines in OpenCV source code before compiling it:

    a. imagwarp.cpp
        > CV_Assert( ssize.area() > 0 );

          When the product of rows and columns of the image to be resized 
          is larger than 2^31, ssize.area() results in a negative number. 
          This appears to be a bug in OpenCV and hopefully will be fixed 
          in the future release. 
          A temporary fix is to build OpenCV with this line commented out.

        > CV_Assert( dsize.area() > 0 );

          For images with height larger than width, it's the above line 
          that causes error. So, the bug in OpenCV.

    Reference: http://stackoverflow.com/questions/31996367/opencv-resize-fails-on-large-image-with-error-215-ssize-area-0-in-funct

2. nolearn:
    Modify definition of predict(self, X) function 
    in .../anaconda/lib/python3.5/site-packages/nolearn/lasagne/base.py as below:

    def predict(self, X):
        if self.regression:
            return self.predict_proba(X)
        else:
            proba  = self.predict_proba(X)
            y_pred = np.empty(proba.shape)
            for i, p in enumerate(proba):
                y_pred[i] = np.divide(p, np.sum(p))
            return y_pred

    This change is required in order to return a vector of probabilistic prediction
    instead hard prediction.


V. USAGE
--------

A. Mac OS:
     ipython image_classification.py -- 
                                     -i <Path for training images>
                                     -o <Path for testing images> 
                                     -x <Path for training csv>

     Note:
	1. Refer ‘./sample/train.csv’ for CSV format of training csv.
	2. Once the script execution finishes, find the predictions in ‘result.csv’.

B. Linux:
    Same as described in (V)-[A].

C. Windows:
    Not tested.


VI. ALGORITHM
-------------

 1. Identify and remove corrupt images.

 2. Preprocess all train images.
        a. Read image from disk.
        b. Transform image colorspace, if specified.
           Note: For the best performance, keep 'RGB' colorspace.
        c. Normalise image data.
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
    with specified hyper-parameters and techniques.

 5. Train a model with preprocessed training images.

 6. Preprocess all test images as specified in (1).

 7. Predict labels for preprocessed testing images 
     using the trained neural network model.

 8. Summarise and save predictions.


VII. MAINTAINER
---------------

Name        Chetan Borse
EMail ID    chetanborse2106@gmail.com
LinkedIn    https://www.linkedin.com/in/chetanrborse
