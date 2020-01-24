# 188.501 Similarity Modeling 1 - WS 2019

## Complete student data
* Johannes Bogensperger - 01427678
* Benjamin Ottersbach - 01576922

## Finding Kermit
The two sub projects in this repository "Kermit_Optical_Recog_VGG16"and "SimModelAudio" are trying to recognize Kermit via images of the muppetshow and with the audio lines. This is a binary problem with two possible states "Kermit is present" and "Kermit is NOT present".

The first project "Kermit_Optical_Recog_VGG16" which is based on a CNN for image recognition, is the final architecture after we tried simple approaches from internet tutorials. For this final architecture, we chose to use the VGG-16 from K. Simonyan and A. Zisserman (Oxford University) proposed in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. This CNN is made for transfer-learning and already trained on various various image like the ImageNet dataset, therefore we only need to train the last layers. For the adaption of this model to our problem we put two dense layer on top of it and a dropout layer to prevent overfitting.

With the second project "SimModelAudio", we experienced far more problems. After following some tutorial for speaker recognition, we came to the conclusion that LSTM classification layers upon the Mel-frequency cepstral coefficients (MFCC) of the audio data is not capable of providing sufficient results. A further learning was that its crucial to reduce the number of coefficients returned by librosa from 20 down to 13/7 since the last coefficient will contain mostly noise. After CNNs didn't deliver the desired results, we chose to use a simple Random Forest as classifier. We had to restrict our hyperparameter searchspace drastically and stroed intermediate MFC coefficients before model training, due to long runtimes.

Since the results were not satisfying with our approaches we tried various sampling techniques and ended up with SMOTE upsampling. We have two possible datasets for learning KERMIT. They can be altering the DATA_DIR1, setting the dataset to learn the classifier upon.

DATA_DIR1 = 'data/kermit/'      # A dataset containing a refined set of very pure samples of Kermits voice without background noize etc.  
DATA_DIR1 = 'data/kermit_big/'  # The full dataset of all audio files initially labelled as "Kermit present" 

## Set Up of the enviroment and Entry point of the code
Our image and audio samples are contained in compressed "data" folders for the projects in the github release "V1.0".
Unzip the files:
* "data_for_audio_recognition_kermit.zip" in SimModelAudio/data/..
* "data_for_visual_rocognition_kermit.zip" in Kermit_Optical_Recog_VGG16/data/..
(don't forget to create the data folder.. empty folders are not added by git and I didn't see the point in adding a .keep file or dummy file..)

and run the corresponding main methods:
* "Kermit_Optical_Recog_VGG16" - Kermit_Optical_Recog_VGG16/main.py
* "SimModelAudio" - SimModelAudio/main.py

Furthermore the image classification keras.model can be found as well in the project folder (models/final_model) and be loaded via keras.models.load_model.

The python environments can be built via the requirements.txt files in each project folder!

## Performance indicators (e.g. Recall, Precision, etc.)

### Image detection
Final results for the test set:

* F1 Score:0.9806835066864783
* Accuracy:0.9887737478411054

| Confusion Matrix  | Predicted Not present | Predicted Present  |
| ----------------: |:---------------------:| -------------:|
| Not present       | 815                 | 8          |
| Present           | 5                  | 330           |

#### ROC Curve

![alt text](https://github.com/bottersb/sm_w19/blob/master/Kermit_Optical_Recog_VGG16/Roc_Curve_VGG16.png =250x)


### Audio detection

#### KERMIT - BIG
The final results for the big and "dirty" dataset which is highly unbalanced and uses all kermit samples (not just pure ones) 

* Accuracy: 0.84762839385018
* F1 Score: 0.4400096176965617

| Confusion Matrix  | Predicted Not present | Predicted Present  |
| ----------------: |:---------------------:| -------------:|
| Not present       | 12041                 | 1072          |
| Present           | 1257                  | 915           |

#### ROC Curve
![alt text](https://github.com/bottersb/sm_w19/blob/master/SimModelAudio/ROC_curve_kermit_big.png)

#### KERMIT - SMALL/PURE
Even though the training sets are always upsampled via SMOTE we couldn't achieve better results on the original highly unbalanced distribution of the test data.

* Accuracy: 0.9537098560354375
* F1 Score: 0.3953712632594021

| Confusion Matrix  | Predicted Not present | Predicted Present  |
| ----------------: |:---------------------:| -------------:|
| Not present       | 12713                 | 412          |
| Present           | 215                  | 205           |


#### ROC Curve
![alt text](https://github.com/bottersb/sm_w19/blob/master/SimModelAudio/ROC_curve_kermit_puresmall.png)

## Timesheets
Our timesheet can be found online:
https://docs.google.com/spreadsheets/d/18DE5sUamwnyQ6VUXzKJwsusqlUtsCt6sKx07ir5BiHI/edit?usp=sharing



