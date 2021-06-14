import cv2
import glob
#import random
import math
import numpy as np
import dlib
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#from shutil import copyfile
import csv
import os
import time

def SortKeys(s):
    return int(os.path.basename(s)[:-4])

emotion = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"] #Emotions used
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat") #Shape predictor
files = glob.glob(r"Fer+ Data Base\\*") #Data base
files.sort(key=SortKeys)
with open('fer2013/fer2013new.csv') as f:
    r = csv.reader(f)
    data = {}
    def data_filters():
        training_data = []
        training_labels = []
        testing_data = []
        testing_labels = []
        sec = 0
        count = 0
        cnt = 0
        train = 0
        test = 0
        neutral = 0
        happy = 0
        #reduce = 0
        #test = 0
        for file, row in zip(files, r):
            if(sec == 0):
                sec = 1
            else:
                #reduce+=1
                #if(reduce%3==0):
                    #continue
                cnt+=1
                if cnt%100 == 0:
                    print(cnt, 'poze in total')
                row = np.array(row)
                idx = np.argmax(row[2:])
                if(idx==0):
                    neutral+=1
                    if(neutral%2==0):
                        continue
                if(idx==1):
                    happy+=1
                    if(happy%2==0):
                        continue
                if(idx>=8):
                    continue
                
                else:
                    image = cv2.imread(file)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
                    clahe_image = clahe.apply(gray)
                    detections = detector(clahe_image, 1)
                    if len(detections)<1:
                        continue
                    else:
                        for k,face in enumerate(detections):
                            shape = predictor(clahe_image, face)
                            x_coord = []
                            y_coord = []
                            for i in range(0,68):
                                x_coord.append(float(shape.part(i).x))
                                y_coord.append(float(shape.part(i).y))
                            xmean = np.mean(x_coord)
                            ymean = np.mean(y_coord)
                            x_center = [(x-xmean) for x in x_coord]
                            y_center = [(y-ymean) for y in y_coord]
                            features = []
                            for x, y, w, z in zip(x_center, y_center, x_coord, y_coord):
                                features.append(w)
                                features.append(z)
                                mean_np = np.asarray((ymean,xmean))
                                coord_np = np.asarray((z,w))
                                distance = np.linalg.norm(coord_np-mean_np)
                                features.append(distance)
                                features.append((math.atan2(y, x)*360)/(2*math.pi))
                                
                            data['features'] = features
                        if(row[0]=='Training'):
                            training_data.append(data['features'])
                            training_labels.append(idx)
                            train+=1
                        else:
                            testing_data.append(data['features'])
                            testing_labels.append(idx)
                            test+=1
                        count+=1
                        if(count%100==0):
                            print(count, 'poze reusite')
        print(train, 'poze pentru antrenare')
        print(test, 'poze pentru testare')
        return training_data, training_labels, testing_data, testing_labels
    
    
    #file = open("Rezultate teste RBF optim.txt","w")
    #file.write("C   GAMMA   ACC\n")
    start = time.time()
    print("Filtering data...")
    training_data, training_labels, testing_data, testing_labels = data_filters()
    """
    training_np = np.array(training_data)
    training_labels_np = np.array(training_labels)
    classifier = SVC(kernel='poly', degree=2, C = pow(2,3), gamma = pow(2,-15), probability=True, tol=1e-3)
    #classifier = SVC(kernel='linear', probability=True, tol=1e-3)
    print("Training SVM with poly kernel ...") 
    classifier.fit(training_np, training_labels_np)
    pickle.dump(classifier, open('fer_poly_degree2_3_-15_no_contempt_fear_disgust.sav','wb'))
    print("Testing model...") 
    testing_np = np.array(testing_data)
    accuracy = classifier.score(testing_np, testing_labels)
    print ("Model accuracy is ", accuracy)
    """
    
    """
    for cost in range(-5,16,2):
        for gamma in range(-15,4,2):
            classifier = SVC(kernel='rbf', C = pow(2,cost), gamma = pow(2,gamma), probability=True, tol=1e-3)#, verbose = True)
            print("training SVM RBF with cost = 2^",cost," and gamma = 2^",gamma) #train SVM
            classifier.fit(training_np, training_labels)
            pickle.dump(classifier, open('model_fer_test_rbf.sav','wb'))
            print("getting accuracies") #Use score() function to get accuracy
            testing_np = np.array(testing_data)
            accuracy = classifier.score(testing_np, testing_labels)
            print ("linear: ", accuracy)
            file.write(str(cost))
            file.write("   ")
            file.write(str(gamma))
            file.write("  ")
            file.write(str(accuracy))
            file.write("\n")
    file.close()
    """
    end = time.time()
    print(end - start)