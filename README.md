# 188.501 Similarity Modeling 1 - WS 2019

## Complete student data
* Johannes Bogensperger - 01427678
* Benni TODO enter Bennies Data

## Finding Kermit
The two sub projects in this repository "Kermit_Optical_Recog_VGG16"and "SimModelAudio" are trying to recognize Kermit via images of the muppetshow and with the audio lines. This is a binary problem with two possible states "Kermit is present" and "Kermit is NOT present".

The first project "Kermit_Optical_Recog_VGG16" which is based on a CNN for image recognition, is the final architecture after we tried simple approaches from internet tutorials. For this final architecture, we chose to use the VGG-16 from K. Simonyan and A. Zisserman (Oxford University) proposed in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. This CNN is made for transfer-learning and already trained on various various image like the ImageNet dataset, therefore we only need to train the last layers.
The network has an Input format of 224x224x3, therefor all our pictures need to be scaled to this format.

With the second project "SimModelAudio", we experienced far more problems. After following some tutorial for speaker recognition, we came to the conclusion that LSTM classification layers upon the Mel-frequency cepstral coefficients (MFCC) of the audio data is not capable of providing sufficient results. A further learning was that its crucial to reduce the number of coefficients returned by librosa from 20 down to 13/7 since the last coefficient will contain mostly noise. After CNNs didn't deliver the desired results, we chose to use a simple Random Forest as classifier. We had to restrict our hyperparameter searchspace drastically and stroed intermediate MFC coefficients before model training, due to long runtimes.

Since the results were not satisfying with our approaches we tried various sampling techniques and ended up with SMOTE upsampling. We have two possible datasets for learning KERMIT. They can be altering the DATA_DIR1, setting the dataset to learn the classifier upon.

DATA_DIR1 = 'data/kermit/'      # A dataset containing a refined set of very pure samples of Kermits voice without background noize etc.
DATA_DIR1 = 'data/kermit_big/'  # The full dataset of all audio files initially labelled as "Kermit present" 

## Set Up of the enviroment and Entry point of the code
Our image and audio samples are contained in compressed "data" folders for the projects in the github release "V1.0".
Unzip the files:
* "data_for_audio_recognition_kermit.zip" in SimModelAudio/data
* "data_for_visual_rocognition_kermit.zip" in Kermit_Optical_Recog_VGG16/data

and run the corresponding main methods:
* "Kermit_Optical_Recog_VGG16" - main.py
* "SimModelAudio" - TODO

## Performance indicators (e.g. Recall, Precision, etc.)


## Timesheets
Our timesheet can be found online:
https://docs.google.com/spreadsheets/d/18DE5sUamwnyQ6VUXzKJwsusqlUtsCt6sKx07ir5BiHI/edit?usp=sharing



