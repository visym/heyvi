"""

# \"Hey Vi!\"

HEYVI is a python package for visual AI that provides systems and trained models for activity detection and object tracking in videos.

HEYVI provides:  

* Real time activity detection of the [MEVA activity classes](https://mevadata.org)
* Real time visual object tracking in long duration videos
* Live streaming of annotated videos to youtube live
* Visual AI from RTSP cameras


# Getting Started

Create a video from a file and track the objects, then create an annotation visualization of the tracked video output (vo)

```python
v = vipy.video.Scene(filename='/path/to/video.mp4').framerate(5)
T = heyvi.system.Tracker()
vo = T(v).annotate('/path/to/annotation.mp4')
```

Create a default RTSP camera and stream the privacy preserving annotated video (e.g. pixelated bounding boxes with captions) to a YouTube live stream.

```python
v = heyvi.sensor.rtsp().framerate(5)
T = heyvi.system.Tracker()
with heyvi.system.YoutubeLive(fps=5, encoder='480p') as s:
     T(v, frame_callback=lambda im: s(im.pixelize().annotate()))
```

## Customization

The following environment varibles may be set by the client to specify live camera streams

VIPY_RTSP_URL='rtsp://user@password:127.0.0.1'    
VIPY_RTSP_URL_0='rtsp://user@password:127.0.0.1'    
VIPY_RTSP_URL_1='rtsp://user@password:127.0.0.2'    
VIPY_YOUTUBE_STREAMKEY='xxxx-xxxx-xxxx-xxxx-xxxx'    
VIPY_CACHE='/home/username/.vipy'    

Where the environment variables VIPY_RTSP_URL_N are the list of cameras that are returned in `heyvi.sensor.cameralist`, and VIPY_RTSP_URL refers to the default RTSP camera in `heyvi.sensor.rtsp`.

Please refer to the [vipy](https://visym.github.io/vipy) documentation for additional environment variables.


## Versioning

To determine what heyvi version you are running you can use:

>>> heyvi.__version__
>>> heyvi.version.is_at_least('0.0.6') 

# Contact

Visym Labs <info@visym.com>

"""

# Import all subpackages
import heyvi.version
import heyvi.system
import heyvi.recognition
import heyvi.detection
import heyvi.sensor

__version__ = heyvi.version.VERSION

