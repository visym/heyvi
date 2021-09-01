import flask
from flask import Flask
import vipy

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello World!"


@app.route("/visualize")
def visualize():
    with open(vipy.visualize.tohtml(vipy.image.owl().annotate()), 'r') as f:
        return f.read()


@app.route("/image")
def image():
    return vipy.image.owl().print().html(alt='owl')


@app.route("/quicklook")
def quicklook():
    return vipy.video.RandomScene().quicklook().print().html(alt='quicklook')


@app.route("/visym/<path:name>")
def send_static(name):
    return flask.send_from_directory('/Users/jebyrne/dev/website', name)


