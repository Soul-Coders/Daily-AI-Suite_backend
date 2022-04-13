import os


from flask import Flask, request, send_file, jsonify
import inference
from PIL import Image
import pytesseract
import cv2
import numpy as np
from numpy import asarray

from keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import model_from_json
#imports for ColorScheme

from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors


#imports for YTranscript

from youtube_transcript_api import YouTubeTranscriptApi
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
import nltk

app = Flask(__name__)

@app.route("/bgremoval", methods=["POST"])
def noBgImage():
    image = request.files['image']   
    image = Image.open(image)   
    # print(type(image))
    im_PIL = image
    image = asarray(image)    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = inference.predict(image)
    im_PIL.putalpha(img)
    im_PIL.save("results/detected.png")


    return send_file('results/detected.png', mimetype='image/png')

@app.route("/ocr", methods=["POST"])
def generate_ocr():
    image = request.files['image']
    image = Image.open(image)
    image = asarray(image)
    image = cv2.resize(image, None, fx=2, fy=2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    config = '--oem 3 --psm 11'
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    txt = pytesseract.image_to_string(image, config = config, lang='eng')
    cleaned_txt = txt.replace('\n\n', ' ').replace('\n', '')

    return jsonify({'text': cleaned_txt})

@app.route("/color", methods=["POST"])
def generate_color():
    image = request.files['image'].read()
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def preprocess(raw):
        image = cv2.resize(raw, (900, 600), interpolation = cv2.INTER_AREA)                                          
        image = image.reshape(image.shape[0]*image.shape[1], 3)
        return image
    
    def rgb_to_hex(rgb_color):
        hex_color = "#"
        for i in rgb_color:
            hex_color += ("{:02x}".format(int(i)))
        return hex_color
    
    def analyze(img):
        clf = KMeans(n_clusters = 5)
        color_labels = clf.fit_predict(img)
        center_colors = clf.cluster_centers_
        counts = Counter(color_labels)
        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
        return hex_colors
    
    modified_image = preprocess(image)
    colorscheme = analyze(modified_image)

    return jsonify({'colors': colorscheme})

@app.route("/summary", methods=["POST"])
def generate_summary():
    data = request.form
    link = data['link']
    
    def get_transcript(videolink):
        video_id = videolink.split("=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ''
        for value in transcript:
            for key,val in value.items():
                if key == 'text':
                    text += val + ' '

        split_text = text.splitlines()
        transcript = " ".join(split_text)
        return transcript
    
    def get_summary(transcript):
        summary = ''
        parser = PlaintextParser.from_string(transcript,Tokenizer("english"))
        summarizer_lsa = LsaSummarizer()
        summary_list =summarizer_lsa(parser.document,1)
        for sentence in summary_list:
            summary += str(sentence)
        return summary
    
    transcript = get_transcript(link)
    summary = get_summary(transcript)

    return jsonify({'summary': summary})

@app.route("/emotion", methods=["POST"])
def mood_songs():
    model = model_from_json(open("model/model.json", "r").read())
    model.load_weights('model/model.h5')
    image = request.files['image'].read()
    # image = Image.open(image)
    # original_img = image
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # image = cv2.imread(image)

    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images = face_haar_cascade.detectMultiScale(gray_image)
    edges = cv2.Canny(image,48,48)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key = cv2.contourArea, reverse=False)
    smallest_item = sorted_contours[0]
    M = cv2.moments(smallest_item)
    x,y,w,h = cv2.boundingRect(smallest_item)
    cv2.rectangle(images, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  2)
    roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
    roi_gray=cv2.resize(roi_gray,(48,48))
    image_pixels = img_to_array(roi_gray)
    image_pixels = np.expand_dims(image_pixels, axis = 0)
    image_pixels /= 255
    predictions = model.predict(image_pixels)
    max_index = np.argmax(predictions[0])
    emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    emotion_prediction = emotion_detection[max_index]

    return jsonify({'mood': emotion_prediction})


# @app.route("/colorizer", methods=["POST"])
# def colorize():
#     colorize = colorizer.Colorizer()
#     image = request.files['image']
#     image = colorize.process_img(image)
#     image = Image.open(image)
#     image.save("results/colorized.png")
#     return send_file('results/colorized.png', mimetype='image/png')

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
