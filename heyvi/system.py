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

    >>> s = heyvi.system.YoutubeLive()
    >>> v = heyvi.sensor.rtsp()                                                                                                                                                                                                                  
    >>> s(v)

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
        return lambda im, v=None: self._vs.write(im if im.shape() == (h,w) else im.resize(height=h, width=w))  # quiet anisotropic resize to stream dimensions

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
    
    """
    def __init__(self, outfile, fps=30):
        assert vipy.util.isvideo(outfile)
        self._vo = vipy.video.Scene(filename=outfile, framerate=fps)
                
    def __repr__(self):
        return '<heyvi.system.Recorder: %s>' % str(self._vo)
    
    def __call__(self, vi, seconds=None, verbose=True):
        assert isinstance(vi, vipy.video.Scene)

        vi = vi if seconds is None else vi.clone().duration(seconds=seconds)
        with self._vo.stream(overwrite=True) as s:
            for (k,im) in enumerate(vi.stream()):
                if verbose:
                    print('[heyvi.system.Recorder][%s][%d]: %s' % (timestamp(), k, im), end='\r')                                    
                s.write(im)                
        return self._vo
                

class Tracker():
    def __init__(self):
        assert vipy.version.is_at_least('1.11.10')
        assert heyvi.version.is_at_least('0.0.5')        
        assert torch.cuda.device_count() >= 4

        objects = ['person', ('car','vehicle'), ('truck','vehicle'), ('bus', 'vehicle'), 'bicycle']  # merge truck/bus/car to vehicle, no motorcycles  
        self._tracker = heyvi.detection.MultiscaleVideoTracker(gpu=[0,1,2,3], batchsize=9, minconf=0.05, trackconf=0.2, maxhistory=5, objects=objects, overlapfrac=0, gate=64, detbatchsize=None)   # TESTING: overlapfrac=6->0
        
    def __call__(self, vi, minconf=0.04, frame_callback=None, verbose=True):
        assert isinstance(vi, vipy.video.Scene)

        for (k, (im,v)) in enumerate(zip(vi.stream().delay(-30), self._tracker(vi, stride=3, continuous=True))):
            print(k)
            if callable(frame_callback) and im is not None:
                frame_callback(im, v)  # FIXME: who owns the annotation
            if verbose and v is not None:
                #print('[heyvi.system.Tracker][%s][%d]: %s' % (timestamp(), k, str(v)+' '*100), end='\r')
                pass
            print((str(im), v))       # FIXME: should include annotations
            
        return vi
    

            
class Actev21():
    def __init__(self):

        assert vipy.version.is_at_least('1.11.10')
        assert heyvi.version.is_at_least('0.0.5')
        assert torch.cuda.device_count() >= 4
        
        self._activitymodel = vipy.downloader.downloadif('https://dl.dropboxusercontent.com/s/ntvjg352b0fwnah/mlfl_v5_epoch_41-step_59279.ckpt',
                                                         vipy.util.tocache('mlfl_v5_epoch_41-step_59279.ckpt'),  # set VIPY_CACHE env 
                                                         sha1='c4457e5b2e4fa1462d552070c47cac9eb2833e47')

        self._annotator = lambda im, f=vipy.image.mutator_show_trackindex_verbonly(confidence=True): f(im).annotate()
        
    def __call__(self, vi, vs=None, minconf=0.04, verbose=True):

        assert isinstance(vi, vipy.video.Scene)
        #assert vi.isloaded() or not vipy.util.isRTSPurl(vi.url()), "Use the Recorder() or buffer an RTSP stream before processing"
        assert vs is None or (isinstance(vs, vipy.video.Stream) and vs.framerate() == 5)
                
        objects = ['person', ('car','vehicle'), ('truck','vehicle'), ('bus', 'vehicle'), 'bicycle']  # merge truck/bus/car to vehicle, no motorcycles
        track = heyvi.detection.MultiscaleVideoTracker(gpu=[0,1,2,3], batchsize=9, minconf=0.05, trackconf=0.2, maxhistory=5, objects=objects, overlapfrac=6, gate=64, detbatchsize=None)
        activities = list(heyvi.label.pip_plus_meva_to_meva.items())
        detect = heyvi.recognition.ActivityTracker(gpus=[0,1,2,3], batchsize=64, modelfile=self._activitymodel, stride=3, activities=activities)   # stride should match tracker stride 4->3
        
        gc.disable()
        (srcdim, srcfps) = (vi.mindim(), vi.framerate())
        vs = vs if vs is not None else contextlib.nullcontext()                
        vi = vi.mindim(960).framerate(5)
        with vs as s:
            for (f, (im,vi)) in enumerate(zip(vi, detect(track(vi, stride=3), mirror=False, trackconf=0.2, minprob=minconf, maxdets=105, avgdets=70, throttle=True, activityiou=0.1))):  # activity detection 
                #if s is not None:
                #    s.write(self._annotator(im).rgb())
                if verbose:
                    print('[heyvi.system.Tracker][%s][%d]: %s' % (timestamp(), f, vi), end='\r')                                    
                
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
    
