[![PyPI version](https://badge.fury.io/py/heyvi.svg)](https://badge.fury.io/py/heyvi)

\"Hey Vi!\"
-------------------

HEYVI: Visym Analytics for Visual AI    
docs: https://developer.visym.com/heyvi

HEYVI is a python package for visual AI that provides systems and trained models for activity detection and object tracking in videos.

HEYVI provides:  

* Real time activity Detection of the [MEVA activity classes](https://mevadata.org)
* Real time visual object tracking in long duration videos
* Live streaming of annotated videos to youtube live
* Visual AI from RTSP cameras


Requirements
-------------------
python >3.6  
[ffmpeg](https://ffmpeg.org/download.html) (required for videos)  
[vipy](https://github.com/visym/vipy), torch, pytorch_lightning (for training)


Installation
-------------------

```python
pip install heyvi
```


Quickstart
-------------------
```python
v = heyvi.sensor.rtsp()
T = heyvi.system.Tracker()
with heyvi.system.YoutubeLive(fps=5, encoder='480p') as s:
     T(v, frame_callback=lambda im, v: s(im.annotate().rgb()))
```

Create a default RTSP camera and GPU enabled object tracker, then stream the annotated video (e.g. bounding boxes with captions) to a YouTube live stream.








