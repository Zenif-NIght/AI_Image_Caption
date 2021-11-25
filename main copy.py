# this program will open an image file from the flickr30k dataseet and run YOLO on the image 
# and create a Conceptual dependency representation fo the image

import cv2
import numpy as np
import os
import pandas as pd

net = cv2.dnn.readNet("YOLO/yolov3.weights", "YOLO/yolov3.cfg")

with open("YOLO/coco.names", "r") as f:
    classes = f.read().splitlines()




if __name__ == "__main__":
    # get the image path
    image_path = './flickr30k_images/flickr30k_images'
    csv_file = './flickr30k_images/results.csv'
    
    # Search the csv file for the image name
    df = pd.read_csv(csv_file, delimiter='|')
    # print(df)
    
    del_img = ''
    
    #iterate through each row of dataframe
    for index, row in df.iterrows():
        print(row.values[0])
        
        image_file = row.values[0]
        comment_number = row.values[1]
        comment = row.values[2]
        
        # get the next image
        img = cv2.imread(os.path.join(image_path, image_file))
        
        
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

        for i in indexes.flatten():
            x, y, w, h = box[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " : " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

        cv2.imshow("Final", img)
        cv2.waitKey(1)
        # if cv2.waitKey(1) & 0xff == ord("q"):
        #     break

    