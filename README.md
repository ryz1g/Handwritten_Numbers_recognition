# Handwritten_Numbers_recognition

This program uses ML to recognize a handwritten number written anywhere on a blank paper with preferable black ink. The algorith is simple, take inpput from the webcam, detect edges, localize and find a number, extract digits and then feed the digits into model for prediction. Please make sure that number iswritten on a plain paper nad there is apt lighting so that noise is not there.

Files included:-

1) trial.py - Contains the code to take input from webcam and print output in console and on screen.
2) train.py - Contains the code used to train the model.
3) helping_functions.py - Contains some coded functions used in processing the image and giving output.
4) model_w.json - Contains the ML model stored in JSON for easy loading.
5) model_w.h5 - Contains weights for the Model.

Update 27/10/2020 - Model weights retrained on processed MNIST and performance improved in further testing. Dealt with a few exceptions.
Update 29/10/2020 - Model weights retrained on canny edges and created input pipeline from the webcam. 
