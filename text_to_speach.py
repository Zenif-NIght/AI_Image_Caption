# this is a simple text to speech converter
# the 1st argument is the text to be converted

import sys
import pyttsx3

def text_to_speech(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 150) 
    engine.say(text)
    engine.runAndWait()
    
    
text_to_speech( sys.argv[1] )