# this program will speak captions from images from the webcam

# this program will open an image file from the flickr30k dataseet and run YOLO on the image 
# and create a Conceptual dependency representation fo the image
import cv2
import numpy as np
import os
import pandas as pd
import pyttsx3
import threading as th

import coco_file as coco

net = cv2.dnn.readNet("YOLO_Weights/yolov3.weights", "YOLO_Weights/yolov3.cfg")

engine = pyttsx3.init()
voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 150) 

def draw_prediction(img):
     # get the image dimensions
        height, width, channels = img.shape
        # create a blob from the image
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output = net.getUnconnectedOutLayersNames()
        layers = net.forward(output)

        box = []
        confidences = []
        class_ids = []
        for out in layers:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    centre_x = int(detection[0] * width)
                    centre_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(centre_x - w / 2)
                    y = int(centre_y - h / 2)

                    box.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = np.array(cv2.dnn.NMSBoxes(box, confidences, 0.5, 0.4))
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(box), 3))

        locations = []
        label_list = []
        for i in indexes.flatten():
            x, y, w, h = box[i]
            label = str(coco.classes[class_ids[i]])
            label_list.append(label)
            locations.append([x, y, w, h])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " : " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

        cv2.imshow("Final", img)
        cv2.waitKey(1)
        
        return [label_list, locations]
        
# find the relative location from location A to location B
def find_relative_location(location_A, location_B, class_name_A,last_class_name_B):
    # tale the magnitude of the distance between the two locations
    delta = np.sqrt((location_A[0] - location_B[0])**2 + (location_A[1] - location_B[1])**2) 
    if delta < 100 :
        return " is near " 
    if location_A[0] < location_B[0]:
        return  " is to the left of " 
    if location_A[0] > location_B[0]:
        return  " is to the right of " 
    if location_A[1] < location_B[1]:
        return  " is above " 
    if location_A[1] > location_B[1]:
        return  " is below " 


# remove classes fro mthe lists
def remove_classes(remove_class, label_list, class_count, locations):
    
    # removethe most occurring class from the class_count dictionary
    del class_count[remove_class]
    # remove the most occurring class from the label list
    new_label_list = []
    new_locations = []
    for i in range(len(label_list)):
        if label_list[i] == remove_class:
            # label_list.pop(i)
            pass
        else:
            new_label_list.append(label_list[i])
            new_locations.append(locations[i])
            
    return new_label_list, class_count, new_locations

# this function pluralizes the conceptual dependency representation
def pluralize(class1,class_count, useThe=False):
    count1 = class_count[class1]
    
    str_1 = " " + class1 
    # add an s is the last class is plural
    if count1 > 1:
        if useThe:
            str_1 = " the " +str(count1) +" "+  class1 + "s"
        else:
            str_1 = " " +str(count1) +" "+ class1 + "s"
    else: 
        if useThe:
            str_1 = "the" + str_1
        else:
            str_1 = 'a' + str_1
    return str_1

# create a recessive function to say the story
def make_recessive_story(label_list,class_count ,locations, story, concept_dep):
    if len(class_count) == 0:
        return story , concept_dep
    elif len(class_count) == 1:
        cur_class = list(class_count.keys())[0]
        if story != "":
            story += " I also see " + pluralize(cur_class,class_count) + "."
        else:
            story += "I see "+ pluralize(cur_class,class_count) + "."
        concept_dep[cur_class] = [coco.lexicon[cur_class], ""]
        label_list,class_count ,locations = remove_classes(cur_class, label_list,class_count ,locations)
        
        return make_recessive_story(label_list, class_count, locations, story, concept_dep)
    elif len(class_count) >= 2:
        # write a sentence about the last 2 classes
              
        # find the least occurring class
        class1 = min(class_count, key=class_count.get)
        # find the most occurring class
        class2 = max(class_count, key=class_count.get)
        if class1 == class2:
            class1 = list(class_count.keys())[0]
            class2 = list(class_count.keys())[1]
                   
        story = story + " I see "+ pluralize(class1,class_count)  + " and " + pluralize(class2,class_count)  + "."
        
        # state the relitive location of class1 and class2
        story = (story + " I note that " +
                    pluralize(class1,class_count, useThe=True) + 
                    find_relative_location(locations[label_list.index(class1)],locations[label_list.index(class2)],class1,class2) 
                    +  pluralize(class2,class_count, useThe=True) + ".")
        
        # construct the concept depencency representation
        # append the relitive location of class1 and class2
        concept_dep[class1] = [coco.lexicon[class1], find_relative_location(locations[label_list.index(class1)],locations[label_list.index(class2)],class1,class2)]
        concept_dep[class2] = [coco.lexicon[class2], find_relative_location(locations[label_list.index(class1)],locations[label_list.index(class2)],class1,class2)]
        
        label_list,class_count ,locations = remove_classes(class1, label_list,class_count ,locations)
        label_list,class_count ,locations = remove_classes(class2, label_list,class_count ,locations)
        return make_recessive_story(label_list, class_count, locations, story, concept_dep)
    else:
        print('THERE IS AN ERROR')
        return 'THERE IS AN ERROR'+ story + "ERROR POINT.", concept_dep



def say_story(story):

    os.system("python3 text_to_speach.py \"" + story+ "\"")   
    # engine.say(story)
    # # wait for the speech to complete but terminate the program
    # engine.runAndWait()
    # # engine.stop()



if __name__ == "__main__":
     

    # open the camera with opencv
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    # every 100 seconds, take a picture the current frame and run the model
    time_count = 0
    
    # main loop
    while True:
        img = cap.read()[1]
        
        # check if its time to run the model
        if time_count % 100 == 0:
            # run the model
            [label_list, locations] = draw_prediction(img)
        
            # if nothering is detected or only one thing is detected
            if len(label_list) <= 0:
                continue
            
            # print a sentence for each image
            print("Robot Image Caption:")
            # count the repted classes in the image
            # create a dictionary with the class and the number of times it appears
            class_count = {}
            for label in label_list:
                if label in class_count:
                    class_count[label] += 1
                else:
                    class_count[label] = 1
            
            print(class_count)
            story = ""
            concept_dep =  {}
            # create story from each class              
            story, concept_dep= make_recessive_story(label_list, class_count, locations, story, concept_dep)
        

            # say the story
            say_story(story)
            
            # print the story
            print(story)
            
            # print the concept depencency representation
            print(concept_dep)
            
        # increment the time count
        time_count += 1
        
        cv2.imshow("Final", img)
        
        
        # exit if the user presses q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        
        


    