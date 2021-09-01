
# Python development web server
# -To run webserver accessible on device
>>> cd /path/to/website
>>> python3 -m http.server
>>> !open https://10.0.1.9:8000


# Flask
pip install flask pyopenssl
cd demo
sh> VIPY_BACKEND=agg FLASK_ENV=development FLASK_APP=visualsearch.py flask run --host=0.0.0.0 --port=8000 
sh> open http://127.0.0.1:/8000


# Speech to text
# brew install portaudio
# pip install pyaudio SpeechRecognition
>>> import speech_recognition as sr
>>> (r,m) = (sr.Recognizer(), sr.Microphone())
>>> with m as source: audio = r.listen(source)
>>> value = r.recognize_google(audio)  # free 

# Speech to text:
# https://www.npmjs.com/package/react-speech-recognition
# https://www.npmjs.com/package/react-speech

# Speech to text in javascript:
# https://dev.to/asaoluelijah/text-to-speech-in-3-lines-of-javascript-b8h
#
# var msg = new SpeechSynthesisUtterance();
# msg.text = "Hello World";
# window.speechSynthesis.speak(msg);
#
# if ('speechSynthesis' in window) {
#  // Speech Synthesis supported 
# }else{
#   // Speech Synthesis Not Supported 
#   alert("Sorry, your browser doesn't support text to speech!");
# }
# https://stackoverflow.com/questions/15653145/using-google-text-to-speech-in-javascript
# https://developers.google.com/web/updates/2013/01/Voice-Driven-Web-Apps-Introduction-to-the-Web-Speech-API


# Text to speech
>>> pip install gTTS
>>> from gtts import gTTS
>>> gTTS(text=value, lang='en', slow=False).save("out.mp3")
  

# Search box:
#  (# Hours, # activities, # sensors) and growing!
# - Hey Vi, how many times did I stand up today?
# - Here is what I know about your standing activities:
# - store everything as a sqllite3 database in memory and regenerate static html files for each activity on demand


# Templated questions:
# what was the last time I ___?
# what day did ___?
# what is the clean score for the house?
# what is the tidy score for the house?
# what is my favorite ___ ?
# what did I do between ___ and ___?
# what did I do today?
# what is the longest time I ___ ?
# what is the shortest time I ___ ?
# when was the last time I ___?
# when was the first time I ___?
# when did I ___ ?
# when did I take off ___ ?
# how many times have I ___?
# how many ___ ?
# who left the ___ out/open?
# who moved the ___?
# where did I put the ___?
# where did I leave the ___?
# where does ___ go to be put away?
# where is the ___?
# did I leave ___ out?
# did I do ___ in the past ___?
# did I do ___ in the last ___?
# did we run out of ___?
# did I forget to ___ in the past ___?
# did anything unusual happen today?
# did anyone ___ ?
# are we out of ___?
# how many ___ did I eat?
# how long did I ___ ?
# how long did it take me to ___?
# how many hours did I spend ___?
# how many calories did I consume today?
# how do I look?
# how frequently do I ___ ?
# remind me to ___?
# alert me if I ___ ?
# show me a dashboard

# when?, how many?, who?, where?


# Sending text notifications
client = boto3.client("sns",aws_access_key_id='',aws_secret_access_key="",region_name="us-east-1")
client.publish(PhoneNumber="+16173353311",Message="Hello World!")    


# Flask Login
https://flask-login.readthedocs.io/en/latest/


# Architecture:
# amazon aws + lambda + cognito + api gateway
# flask running in lambda for RESTful web services and dynamic web pages
# sensors communicate with inference platform (shared GPU resources)
# inference platform communicates with REST API via API gateway to POST/PUT all detections
# the inference platform should have abstracted classes for communicating with API gateway
# the inference platform abstraction should allow for inference node to be anywhere (cloud, on-premises)
# mobile device frontend communicates with REST API via API gateway to GET queries for on-device aggregation, or to serve web pages
# mobile device will install sensor (show QR code to sensor) to get it authenticated on wifi and communicating with backend
# the mobile device can make an API request for notification, which when received by lambda, will trigger push notification (backend needs to decrypt)

