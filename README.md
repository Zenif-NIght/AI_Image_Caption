# AI_Image_Caption

## Description

## Installation
Go to this box and download the YOLO model directory and place it in the same directory as this file.
https://usu.box.com/s/hlzi8azhuhdwbiqw0kzu2bykppe5ql6d


use pip to install the packages
all the required packages are in requirements.txt

simply run the following command

```
pip install -r requirements.txt
```
if on ubuntu:
```
sudo apt install espeak
```

# Usage
You can run the code by running the following command
```
python3 main.py
```

note I have a bug where the 1st image does not show up.also if you don't hear the speech turn on your speakers

# Results
The results are shown in the in the output csv file: results_out.csv. The first column is the image name and the second column is the conceptual dependency representation. the las 2 columns are the robot_caption and human_caption. 

Also note the CSV is delimited by a '|'. 

### Example Outpu from the csv file:
```
2331827016.jpg|{'person': ['ACTOR', ' is to the right of '], 'horse': ['ACTOR', ' is to the right of ']}| I see a person and a horse. I note that the person is to the right of the horse.| Rider riding horse as it jumps over a fence .
```

