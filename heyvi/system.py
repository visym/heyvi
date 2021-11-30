import os
import vipy
import gc
import torch
import heyvi.recognition
import heyvi.detection
import heyvi.version
import heyvi.label
import contextlib
import itertools
from heyvi.util import timestamp


class YoutubeLive():
    """Youtube Live stream.

    >>> Y = heyvi.system.YoutubeLive(encoder='480p')
    >>> v = heyvi.sensor.rtsp()                                                                                                                                                                                                                  
    >>> Y(v)

    Args:
        encoder [str]['480p, '720p', '360p']:  The encoder settings for the youtube live stream
        fps [float]:  The framerate in frames per second of the output stream.  
        streamkey [str]:  The youtube live key (https://support.google.com/youtube/answer/9854503?hl=en), or set as envronment variable VIPY_YOUTUBE_STREAMKEY

    """
    
    def __init__(self, streamkey=None, url='rtmp://a.rtmp.youtube.com/live2', fps=30, encoder='480p'):
        assert streamkey is not None or 'VIPY_YOUTUBE_STREAMKEY' in os.environ
        streamkey = streamkey if streamkey is not None else os.environ['VIPY_YOUTUBE_STREAMKEY']
        
        # https://support.google.com/youtube/answer/2853702?hl=en#zippy=%2Cp
        self._encoder_recommended = {'720p':{'width':1280, 'height':720, 'bitrate': '4000k'},
                                     '480p':{'width':854, 'height':480, 'bitrate': '1000k'},
                                     '360p':{'width':640, 'height':360, 'bitrate': '1000k'}}
        
        assert encoder in self._encoder_recommended
        self._encoder = self._encoder_recommended[encoder]

        self._url = '%s/%s' % (url, streamkey)
        assert vipy.util.isurl(self._url)
        self._vo = vipy.video.Scene(url=self._url, framerate=fps)
        
    def __repr__(self):
        return '<heyvi.system.YoutubeLive: url=%s, framerate=%2.1f>' % (str(self._vo.url()), self._vo.framerate())

    def __enter__(self):
        (h,w,br) = (self._encoder['height'], self._encoder['width'], self._encoder['bitrate'])        
        self._vs = self._vo.stream(write=True, bitrate=br)
        return lambda im: self._vs.write(im.rgb() if im.shape() == (h,w) else im.rgb().resize(height=h, width=w))  # quiet anisotropic resize to stream dimensions

    def __exit__(self, type, value, tb):
        self._vs.__exit__(type, value, tb)
    
    def __call__(self, vi, verbose=True):
        assert isinstance(vi, vipy.video.Scene)

        (h,w,fps) = (self._encoder['height'], self._encoder['width'], self._vo.framerate())
        with self as s:
            for (k,im) in enumerate(vi.framerate(fps).resize(height=h, width=w)):
                if verbose:
                    print('[heyvi.system.YoutubeLive][%s][%d]: %s' % (timestamp(), k,im), end='\r')
                s(im)  # write frame to live stream
        return self

    
class Recorder():
    """Record a livestream to an output video file
    
    This will record an out streaming to the provided outfile

    >>> v = vipy.video.Scene(url='rtsp://...', framerate=30)
    >>> R = Recorder('/tmp/out.mp4', framerate=5)
    >>> R(v, seconds=60*60)

    To buffer to memory, you do not need this recorder, use (for small durations):

    >>> v = v.duration(seconds=3).load().saveas('/tmp/out.mp4')

    This will record three seconds from the provided RTSP stream and save in the usual way to the output file

    To record frame by frame:

    >>> v = vipy.video.RandomScene()
    >>> with Recorder('out.mp4') as r:
    >>>    for im in v:
    >>>        r(im.annotate().rgb())  # write individual frames from video v

    """
    def __init__(self, outfile, fps=30, overwrite=False):
        assert vipy.util.isvideo(outfile)
        self._vo = vipy.video.Scene(filename=outfile, framerate=fps)
        self._overwrite = overwrite
        
    def __enter__(self):
        self._vs = self._vo.stream(write=True, overwrite=self._overwrite)
        return lambda im: self._vs.write(im.rgb())  

    def __exit__(self, type, value, tb):
        self._vs.__exit__(type, value, tb)
        
    def __repr__(self):
        return '<heyvi.system.Recorder: %s>' % str(self._vo)
    
    def __call__(self, vi, seconds=None, verbose=True):
        assert isinstance(vi, vipy.video.Scene)

        vi = vi if seconds is None else vi.clone().duration(seconds=seconds)
        vi = vi.framerate(self._vo.framerate())
        with self._vo.stream(overwrite=True) as s:
            for (k,im) in enumerate(vi.stream()):
                if verbose:
                    print('[heyvi.system.Recorder][%s][%d]: %s' % (timestamp(), k, im), end='\r')                                    
                s.write(im)                
        return self._vo
                

class Tracker():
    """heyvi.system.Tracker class

    To run on a livestream:

    ```python
    v = heyvi.sensor.rtsp()
    T = heyvi.system.Tracker()
    with heyvi.system.YoutubeLive(fps=5, encoder='480p') as s:
        T(v, frame_callback=lambda im: s(im.pixelize().annotate(fontsize=15, timestamp=heyvi.util.timestamp(), timestampoffset=(6,10))), minconf=0.5)
    ```

    To run on an input file as a batch:

    ```python
    v = vipy.video.Scene(filename=/path/to/infile.mp4', framerate=5)
    T = heyvi.system.Tracker()
    v_tracked = T(v)
    v_tracked.annotate('annotation.mp4')    
    ```

    To stream tracks computed per frame

    ```python
    vi = vipy.video.Scene(filename=/path/to/infile.mp4', framerate=5)
    for (f,vo) in enumerate(T.stream(vi)):
        print(vo)  # tracking result at frame f
    ```

    To stream tracks computed per frame, along with pixels for current frame

    ```python
    vi = vipy.video.Scene(filename=/path/to/infile.mp4', framerate=5)
    for (f,(im,vo)) in enumerate(zip(vi, T.stream(vi)))
        print(vo)  # tracking result at frame f
        print(im)  # `vipy.image.Image` with pixels available as im.numpy()
    ```

    To stream tracks computed per frame, along with the most recent video clip of length 16:

    ```python
    vi = vipy.video.Scene(filename=/path/to/infile.mp4', framerate=5)
    for (f,(vc,vo)) in enumerate(zip(vi.stream().clip(16), T.stream(vi)))
        print(vo)  # tracking result at frame f
        print(vc)  # `vipy.video.Scene` with pixels for clips of length 16
    ```

    For additional use cases for streaming batches, clips, frames, delays see the [vipy documentation](https://visym.github.io/vipy)

    Returns:
        `vipy.video.Scene` objects with tracks corresponding to objects in `heyvi.detection.MultiscaleVideoTracker.classlist`.  Object tracks are "person", "vehicle", "bicycle".

    """
    def __init__(self):
        assert vipy.version.is_at_least('1.11.11')
        assert heyvi.version.is_at_least('0.0.5')        
        assert torch.cuda.device_count() >= 4

        objects = ['person', ('car','vehicle'), ('truck','vehicle'), ('bus', 'vehicle'), 'bicycle']  # merge truck/bus/car to vehicle 
        self._tracker = heyvi.detection.MultiscaleVideoTracker(gpu=[0,1,2,3], batchsize=9, minconf=0.05, trackconf=0.2, maxhistory=5, objects=objects, overlapfrac=0, gate=64, detbatchsize=None)  
        
    def __call__(self, vi, frame_callback=None, verbose=True):
        """Batch tracking of input video file"""
        assert isinstance(vi, vipy.video.Scene)

        for (k, (im,v)) in enumerate(zip(vi.stream(buffered=True).frame(delay=5), self._tracker(vi, stride=3, buffered=vi.islive()))):
            if callable(frame_callback) and im is not None:
                frame_callback(im)  
            if verbose and v is not None:
                print('[heyvi.system.Tracker][%s][%d]: %s' % (timestamp(), k, str(v)+' '*100), end='\r')
        return vi
    
    def stream(self, vi):
        """Tracking iterator of input video"""        
        for (k, (im,v)) in enumerate(zip(vi.stream(buffered=True).frame(delay=5), self._tracker(vi, stride=3, buffered=vi.islive()))):
            yield v

            
class Actev21():
    """heyvi.system.Actev21 class

    Real time activity detection for the 37 MEVA (https://mevadata.org) activity classes

    >>> v = heyvi.sensor.rtsp().framerate(5)
    >>> S = heyvi.system.Actev21()
    >>> with heyvi.system.YoutubeLive(fps=5, encoder='480p') as s:
    >>>     S(v, frame_callback=lambda im, imraw, v: s(im), minconf=0.2)

    """
    
    def __init__(self):

        assert vipy.version.is_at_least('1.11.11')
        assert heyvi.version.is_at_least('0.0.5')
        assert torch.cuda.device_count() >= 4
        
        self._activitymodel = vipy.downloader.downloadif('https://dl.dropboxusercontent.com/s/ntvjg352b0fwnah/mlfl_v5_epoch_41-step_59279.ckpt',
                                                         vipy.util.tocache('mlfl_v5_epoch_41-step_59279.ckpt'),  # set VIPY_CACHE env 
                                                         sha1='c4457e5b2e4fa1462d552070c47cac9eb2833e47')

        self._annotator = lambda im, f=vipy.image.mutator_show_trackindex_verbonly(confidence=True): f(im).annotate(timestamp=heyvi.util.timestamp(), timestampoffset=(6,10), fontsize=15).rgb()
        
    def __call__(self, vi, vs=None, minconf=0.04, verbose=True, frame_callback=None, livestream=False):

        assert isinstance(vi, vipy.video.Scene)
        assert vs is None or (isinstance(vs, vipy.video.Stream) and vs.framerate() == 5)

        livedelay = 2*15*5 if vi.islive() or livestream else 5 
        objects = ['person', ('car','vehicle'), ('truck','vehicle'), ('bus', 'vehicle'), 'bicycle']  # merge truck/bus/car to vehicle, no motorcycles
        track = heyvi.detection.MultiscaleVideoTracker(gpu=[0,1,2,3], batchsize=9, minconf=0.05, trackconf=0.2, maxhistory=5, objects=objects, overlapfrac=6, gate=64, detbatchsize=None)
        activities = list(heyvi.label.pip_plus_meva_to_meva.items())
        detect = heyvi.recognition.ActivityTracker(gpus=[0,1,2,3], batchsize=64, modelfile=self._activitymodel, stride=3, activities=activities)   # stride should match tracker stride 4->3
        
        gc.disable()
        (srcdim, srcfps) = (vi.mindim(), vi.framerate())
        vs = vs if vs is not None else contextlib.nullcontext()                
        vi = vi.mindim(960).framerate(5)
        for (f, (im,vi)) in enumerate(zip(vi.stream(buffered=True).frame(delay=livedelay),  # live stream delay (must be >= 2x finalized period)
                                          detect(track(vi, stride=3, buffered=vi.islive()),
                                                 mirror=False, trackconf=0.2, minprob=minconf, maxdets=105, avgdets=70, throttle=True, activityiou=0.1, buffered=vi.islive(), finalized=(livedelay//2) if vi.islive() or livestream else True))):
            if callable(frame_callback) and im is not None:
                frame_callback(self._annotator(im.clone()), im, vi)  
            if verbose:
                print('[heyvi.system.Actev21][%s][%d]: %s' % (timestamp(), f, vi), end='\r')                                    
                
        vi.activityfilter(lambda a: a.category() not in ['person', 'person_walks', 'vehicle', 'car_moves'])   # remove background activities
        vo = vi.framerate(srcfps)  # upsample tracks/activities back to source framerate
        vo = vo.mindim(srcdim)  # upscale tracks back to source resolution
        gc.enable()

        return vo


    def annotate(self, v, outfile, minconf=0.1, trackonly=False, nounonly=False, mindim=512):
        return (v.mindim(mindim).activityfilter(lambda a: a.confidence() >= float(minconf))
                .annotate(mutator=vipy.image.mutator_show_trackindex_verbonly(confidence=True) if (not trackonly and not nounonly) else (vipy.image.mutator_show_trackonly() if trackonly else vipy.image.mutator_show_nounonly(nocaption=True)),
                          timestamp=True,
                          fontsize=6,
                          outfile=outfile))  # colored boxes by track id, activity captions with confidence, 5Hz, 512x(-1) resolution    
    


    
