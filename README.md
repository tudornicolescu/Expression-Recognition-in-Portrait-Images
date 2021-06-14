# Expression-Recognition-in-Portrait-Images
The purpose of the project was to develop an application that can detect and identify the facial expressions of the user. At first, the face is identified and a series of key points are detected, according to the FACS model. The machine learning model is trained using a support-vector machine, with the key points as the main features. A data based with the necessary descriptions is built and used for training and testing. The application is built around a graphical user interface. The interface is optimized to predict the facial expression in real time.

The database used for training can be found here:
https://github.com/microsoft/FERPlus
@inproceedings{BarsoumICMI2016,
    title={Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution},
    author={Barsoum, Emad and Zhang, Cha and Canton Ferrer, Cristian and Zhang, Zhengyou},
    booktitle={ACM International Conference on Multimodal Interaction (ICMI)},
    year={2016}
}

The file shape_predictor_68_face_landmarks.dat can be downloaded from: https://github.com/davisking/dlib-models
