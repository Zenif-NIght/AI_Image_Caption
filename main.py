# this program will open an image file from the flickr30k dataseet and run YOLO on the image 
# and create a Conceptual dependency representation fo the image

import cv2
import numpy as np
import os
import pandas as pd

import coco_file as coco

net = cv2.dnn.readNet("YOLO/yolov3.weights", "YOLO/yolov3.cfg")



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
        
# find the erelitve location from location A to location B
def find_relative_location(location_A, location_B, class_name,last_class_name):
    if np.max(np.array(location_B[0:2]) - np.array(location_A[0:2])) < 100 :
        return class_name +" IS THE NEAR " +last_class_name
    if location_B[0] < location_A[0]:
        return class_name + " IS TO THE LEFT OF " + last_class_name
    if location_B[0] > location_A[0]:
        return class_name + " IS TO THE RIGHT OF " + last_class_name
    if location_B[1] < location_A[1]:
        return class_name + " IS ABOVE " + last_class_name
    if location_B[1] > location_A[1]:
        return class_name + " IS BELOW " + last_class_name

# tis function will take a list of labels and locations and create a conceptual dependency representation recursively
def create_concept_dep(label_list, locations):
    concept_dep = {}
    if len(label_list) == 0:
        return []
    if len(label_list) == 1:
        return [label_list[0]]
    else:
        last_class_name = label_list[-1]
        last_location = locations[-1]
        for i in range(len(label_list)):
            concept = label_list[i]
            locations = locations[i]
            class_name = label_list[i]
            if concept in coco.lexicon:
                concept_dep[concept] = [coco.lexicon[concept], find_relative_location(locations ,last_location,class_name,last_class_name)]
                last_class_name = class_name
                last_location = locations
            else:
                concept_dep[concept]= create_concept_dep(label_list[i:], locations[i:])
                
        return concept_dep


if __name__ == "__main__":
    # get the image path
    image_path = './flickr30k_images/flickr30k_images'
    csv_file = './flickr30k_images/results.csv'
    csv_out_file = './results_out.csv'
    # Search the csv file for the image name
    df = pd.read_csv(csv_file, delimiter='|')
    # print(df)
    
    out_df = pd.DataFrame(columns=['image_name', 'concept_dep' ,'robot_caption', 'human_caption'])
        
    cont = 0
    # loop over the image in the folder
    for image_name in os.listdir(image_path):
        
        # find the dataframe row with the image name
        data = df[df['image_name'] == image_name]

        if data.empty:
            os.remove(image_path +'/'+ image_name)
            continue
        
        # get the image caption
        print("__________STARTING-IMG \"", image_name ,"\"_________________")
        print("Human Image Captions ")
        print(data.values[0][2])
        print(data.values[1][2])
        print(data.values[2][2])
        print(data.values[3][2])

        # get the next image
        img = cv2.imread(os.path.join(image_path, image_name))
        
        [label_list, locations] = draw_prediction(img)
        
        if len(label_list) == 0:
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
        
        # create story from each class
        story = ""
        
        # concept_dep = create_concept_dep(label_list, locations)
        concept_dep = {}
        
        # find the least occurring class
        least_occurring_class = min(class_count, key=class_count.get)
        
        # add the least occurring class to the story
        story += "I saw "+ str(class_count[least_occurring_class]) +" "+ least_occurring_class + " "
        
        # find the most occurring class
        most_occurring_class = max(class_count, key=class_count.get)
        
        # add the most occurring class to the story
        story += "and " + str(class_count[most_occurring_class]) + " "+ most_occurring_class + "."
        
        # describe relation between the most and least occurring class
        story += " The " + find_relative_location(
                        locations[label_list.index(most_occurring_class)], 
                        locations[label_list.index(least_occurring_class)], 
                        most_occurring_class, 
                        least_occurring_class) + "."
        
        
        # print the story
        print(story)
        
        # add the story to the dataframe
        out_df = out_df.append({'image_name': image_name, 'concept_dep' : concept_dep, 'robot_caption': story, 'human_caption': data.values[0][2]}, ignore_index=True)
        
        # break
        cont += 1
        if cont == 10:
            break
    
    # create a new csv file with the story
    out_df.to_csv(csv_out_file, sep='|', index=False)
        
    # print("__________ENDING-IMG \"", image_name ,"\"_________________")
    # print(class_count)
    
    # print(coco_lexicon['person'])
        
        


    