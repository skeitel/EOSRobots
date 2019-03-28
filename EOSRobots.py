"""
THIS PROGRAM USES IMAGE RECOGNITION THROUGH KERAS AND OPENCV TO

1) USING THE WEBCAM, READ IMAGE OF USER'S FACE AND DETECT GENDER AND AGE
2) DETECTION OF SMILE AND PERCENTAGE OF TIME SMILE IS PRESENT
3) DEPENDING ON % OF TIME WITH PRESENCE OF SMILE, THE PROGRAM WILL SUGGEST "HAPPY ACTIONS"
OR SEND A VOICE-TAILORED PRE-POPULATED TXT TO A FRIEND
4) LISTENS TO INPUT BY USER AND REPEATS WORDS SAID OUTLOUD
5) MAKES AN INTERNET SEARCH BASED ON USER'S AUDIO INPUT AND OPENS A NEW BROWSWER WINDOW WITH THE SEARCH RESULTS

This program is based on using Python to fine-tuning and add functionality to some of the following tutorials/techniques:
https://www.youtube.com/watch?v=atJmJ8tNc3U&list=PLZoTAELRMXVNUcr7osiU7CCm8hcaqSzGw&index=26
https://www.youtube.com/watch?v=671zrErshRY
https://www.youtube.com/watch?v=guNm3IATvBA
javiermarti.co.uk
"""
#TO DO
#things I know about you
#read an article
#check the weather

'''
#FACE DETECTION BEGINS ###############################
'''
#import robot face
# import robot_img
# print(robot_img.img)

#import libraries
import cv2
import os
from time import sleep
import numpy as np
import argparse
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file


class FaceCV(object):


    """
    Singleton class for face recongnition task
    """
    # CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"
    # WRN_WEIGHTS_PATH = ".\\pretrained_models\\weights.18-4.06.hdf5"
    CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = ".\\pretrained_models\\weights.18-4.06.hdf5"


    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def detect_face(self):

        age_gender_info = []

        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(0)
        # infinite loop, break by key ESC
        for el in range(100):
        #while True:
            if not video_capture.isOpened():
                sleep(5)
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(self.face_size, self.face_size)
            )



            if faces is not ():

                # placeholder for cropped faces
                face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
                for i, face in enumerate(faces):
                    face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                    (x, y, w, h) = cropped
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                    face_imgs[i,:,:,:] = face_img

                if len(face_imgs) > 0:
                    # predict ages and genders of the detected faces
                    results = self.model.predict(face_imgs)
                    predicted_genders = results[0]
                    ages = np.arange(0, 101).reshape(101, 1)
                    predicted_ages = results[1].dot(ages).flatten()

  ################# draw results

                for i, face in enumerate(faces):
                    label = "{},{}".format(int(predicted_ages[i]),
                                            "Female" if predicted_genders[i][0] > 0.5 else "Male")

                    self.draw_label(frame, (face[0], face[1]), label)
                    #PRINT identification DATA ON CONSOLE
                    print('FACE DETECTED')
                    print('You are a',label.split(',')[1].lower(),'and around',label.split(',')[0],'years old')
                    age_gender_info.append(int(label.split(',')[0]))
                    print(age_gender_info)
            else:
                print('No faces detected. Please approach camera...')

            cv2.imshow('Keras Faces', frame)
            if cv2.waitKey(5) == 27:  # ESC key press
                break
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()
        #print(age_gender_info)
        age_list = np.array(age_gender_info)
        your_age = np.sum(age_list) / age_list.shape
        #print('You are between', np.min(age_list), 'and', np.max(age_list), '...so I would guess you are about', int(your_age), 'years old?')
        print('*' * 100)
        print('I THINK YOU ARE', int(your_age), 'YEARS OLD')
        print('*'*100)

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    depth = args.depth
    width = args.width

    face = FaceCV(depth=depth, width=width)

    face.detect_face()

if __name__ == "__main__":
    main()


'''
# SMILE DETECTION BEGINS ##############################
'''
print('*'*100)
print('SMILE DETECTION IS STARTING...SHOW ME A GOOD OPEN SMILE NOW AND I WILL SEE THAT YOU ARE SMILING!')
print("Let's see how happy you are today")
print('*'*100)

# detect number of smiles
smiles_counter = []

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining a function that will do the detections
def detect(gray, frame):

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 18)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 18)


        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            print('Subject is CURRENTLY SMILING ***')
            smiles_counter.append('x')

        # else:
        #     print('\nSubject is NEUTRAL')

    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)

range_lenght = 100
for el in range(range_lenght):
#while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)

    if cv2.waitKey(5) == 27:  # ESC key press
        break
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
video_capture.release()
cv2.destroyAllWindows()

#quantify smiling time
percent_smiling = int(len(smiles_counter)) * int(range_lenght) / 100
print('*' * 100)
print('Subject was smiling for ', percent_smiling , '% of the interaction')
print('*' * 100)

#Quantify smiling and start actions
print('*'*100)
if percent_smiling <= 15:
    print('*' * 100)
    print('You do not seem very happy today...')
    print('*' * 100)
    permission = input('Would you like to activate your favourite "Happy feelings" actions?\n')
    print('*' * 100)
    if 'Yes' or 'yes' or 'y' 'yeah' or 'Y' or 'Yeah' in permission:
        print('CONNECTING TO THE INTERNET OF THINGS')
        print('> OPENING BLINDS 10% MORE...')
        print('> RELEASING LAVENDER AIR FRAGRANCE...')
        print('> CHANGING TO UPBEAT DESKTOP WALLPAPER...')
        print('> STARTING TO PLAY FAVOURITE MOZART MUSIC...')
        print('> TEXTING FRIEND X TO GO TO DO ACTIVITY Y LATER')
    else:
        pass

else:
    print('*' * 100)
    print('You look pretty happy today! Would you like me to text friend X to arrange a nice catchup?\n')
    friend_name = input('Who should I text?\n')
    place = input('What place do you want to suggest to go together to?\n')
    print('*' * 100)
    print('Sending text to ', friend_name, 'now ...')
    print('...')
    text = 'Hey ' + friend_name + '! would you like to come with me to ' + place + ' today?'
    print('> Text "' + text + '" has been sent!')
    print('> Awaiting response from ' + friend_name)
    print('*' * 100)


###########################################################
#THINGS I KNOW ABOUT YOU ##################################
###########################################################
# print("By the way...I don't know you much, but I am going to make some wild guesses about you...")
# if label.split(',')[1].lower() > 30 and label.split(',')[1].lower() <= 40:
#     print("You are able to relate to most people, and capable of effectively managing your time, energy and emotions, most of the time.\nYou feel you are still young, energetic and sharp. You are figuring out the cheat codes for life...and it feels good.\nYou feel wiser and smarter than your previous self. You are thinking more about the future...but mostly enjoying the present.\nAlthough you are still interested in attracting others to you, you care to buy things that will last and are functional, rather than just flashy objects."
# )
# if label.split(',')[1].lower() >= 20 and label.split(',')[1].lower() <= 30:
#     print("You are increasingly aware of the ticking of your biological clock.\nYou no longer care about a lot of things you cared about in your 20s, or care so much for the approval of your girlfriends. Indeed, younger girls seem pretty immature to you now!\nYou are starting to appreciate and seek stability and security in relationships.\nYou don’t know the most popular songs or pop idols anymore!\nYou also have a better understanding of how to deal with your hair and how to better care for your skin...and appreciate more the time you get to relax and enjoy by yourself." )



#BEGINS SPEECH RECOGNIZER #################################
###########################################################
from gtts import gTTS
import pyglet
import time, os

#define function to save and play audio file
def tts(text, lang):
    file = gTTS(text = text, lang = lang)
    filename = 'temp.mp3'
    file.save(filename)

    music = pyglet.media.load(filename)
    music.play()

    time.sleep(music.duration)
    #os.remove(filename) #use to remove file after each recording


print('*' * 100)
print('STARTING AUDIO CAPABILITIES DEMONSTRATION...')
print('*' * 100)
tts('Welcome, I am Laura the robot. Javier Marti has programmed me to hear what you say and read it back to you. Now I am going to show you this, so be ready and just say something when you see the prompt on the screen in a few seconds...', lang = 'en')



import speech_recognition as sr
#https://www.youtube.com/watch?v=guNm3IATvBA
r = sr.Recognizer()
with sr.Microphone() as source:
    print('*' * 100)
    print('\n *** Listening. Say anything you want outloud, right now... ***')
    audio = r.listen(source)
    print('*' * 100)

    try:
        text = r.recognize_google(audio).capitalize()
        print(f'\nYou just said:\n\n *** "{text}"  *** ')

        lang = 'en'
        #lang1 = 'es-ES'
        tts(text, lang)

    except Exception as e:
        print('Sorry, could not recognize your voice', e)


#BEGINS GOOGLE SEARCH CAPABILITY ##########################
###########################################################
tts('I can also do an Internet search', lang)
tts('Just say outloud in two seconds, what you want me to search for you', lang)

with sr.Microphone() as source:
    print('*' * 100)
    print(' *** Listening. Say what you want to search for outloud, right now... ***')
    audio = r.listen(source)
    print('*' * 100)

    try:
        text_to_search = r.recognize_google(audio).capitalize()
        print(f'\nTHE TEXT YOU WANT ME TO SEARCH IS:\n\n *** "{text_to_search}"  *** ')
        tts('The text you want me to search for is', lang)

        lang = 'en'
        #lang1 = 'es-ES'
        tts(text_to_search, lang)

    except Exception as e:
        print('Sorry, could not recognize your voice', e)



#Begins to open browser
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Create a new instance of the Firefox driver
driver = webdriver.Firefox()
# go to the google home page
driver.get("http://www.google.com")
# the page is ajaxy so the title is originally this:
print(driver.title)
# find the element that's name attribute is q (the google search box)
inputElement = driver.find_element_by_name("q")
# type in the search
inputElement.send_keys(text_to_search)
# submit the form (although google automatically searches now without submitting)
inputElement.submit()
try:
    # we have to wait for the page to refresh, the last thing that seems to be updated is the title
    WebDriverWait(driver, 10).until(EC.title_contains("cheese!"))
    # You should see "cheese! - Google Search"
    #print(driver.title)
finally:
    pass
#    driver.quit()