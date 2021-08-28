import os
import vipy
import gc
import torch
import heyvi.recognition
import heyvi.detection
import pycollector.version
import pycollector.label
import heyvi.version
import heyvi.label
import contextlib


class YoutubeLive():
    """Youtube Live stream"""
    
    def __init__(self, streamkey, url='rtmp://a.rtmp.youtube.com/live2', fps=30):
        self._url = '%s/%s' % (url, streamkey)
        assert vipy.util.isurl(self._url)
        self._vo = vipy.video.Scene(url=self._url, framerate=fps)

    def __repr__(self):
        return '<heyvi.system.YoutubeLive: %s>' % str(self._vo)
    
    def __call__(self, vi):
        assert isinstance(vi, vipy.video.Scene)
        
        with self._vo.stream(write=True) as s:
            for (k,im) in enumerate(vi.stream()):
                print(k,im)
                s.write(im)

                
class Recorder():
    """Record a livestream to an output video file
    
    This will record an out streaming to the provided outfile

    >>> R = Recorder('/tmp/out.mp4', framerate=5)
    >>> R(vipy.video.Scene(url='rtsp://...', framerate=30), seconds=60*60)

    To buffer to memory, you do not need this recorder, use (for small durations):

    >>> v = vipy.video.Scene(url='rtsp://...', framerate=30).duration(seconds=3).load()

    This will record three seconds from the provided RTSP stream.
    """
    def __init__(self, outfile, fps=30, seconds=None):
        assert vipy.util.isvideo(outfile)
        self._vo = vipy.video.Scene(filename=outfile, framerate=fps)
        
    def __repr__(self):
        return '<heyvi.system.Recorder: %s>' % str(self._vo)
    
    def __call__(self, vi, seconds=None):
        assert isinstance(vi, vipy.video.Scene)

        vi = vi if seconds is None else vi.clone().clip(0, self._vo.framerate()*seconds)
        with self._vo.stream(overwrite=True) as s:
            for (k,im) in enumerate(vi.stream()):
                print('%s, frame=%d' % (str(im), k))
                s.write(im)                
        return self._vo
                
                
class Actev21():
    def __init__(self):

        assert vipy.version.is_exactly('1.11.7')
        assert heyvi.version.is_exactly('0.0.3')
        assert torch.cuda.device_count() >= 4
        
        self._activitymodel = vipy.downloader.downloadif('https://dl.dropboxusercontent.com/s/ntvjg352b0fwnah/mlfl_v5_epoch_41-step_59279.ckpt',
                                                         vipy.util.tocache('mlfl_v5_epoch_41-step_59279.ckpt'),  # set VIPY_CACHE env 
                                                         sha1='c4457e5b2e4fa1462d552070c47cac9eb2833e47')

        self._annotator = lambda im, f=vipy.image.mutator_show_trackindex_verbonly(confidence=True): f(im).annotate()
        
    def __call__(self, vi, vs=None, minconf=0.04):

        assert isinstance(vi, vipy.video.Scene)
        assert vi.isloaded() or not vipy.util.isRTSPurl(vi.url())
        assert vs is None or isinstance(vs, vipy.video.Stream)
        vs = vs if vs is not None else contextlib.nullcontext()        
        
        objects = ['person', ('car','vehicle'), ('truck','vehicle'), ('bus', 'vehicle'), 'bicycle']  # merge truck/bus/car to vehicle, no motorcycles
        track = heyvi.detection.MultiscaleVideoTracker(gpu=[0,1,2,3], batchsize=9, minconf=0.05, trackconf=0.2, maxhistory=5, objects=objects, overlapfrac=6, gate=64, detbatchsize=None)
        activities = list(heyvi.label.pip_plus_meva_to_meva.items())
        detect = heyvi.recognition.ActivityTracker(gpus=[0,1,2,3], batchsize=64, modelfile=self._activitymodel, stride=3, activities=activities)   # stride should match tracker stride 4->3
        
        gc.disable()
        (srcdim, srcfps) = (vi.mindim(), vi.framerate())
        vi = vi.mindim(960).framerate(5)
        with vs as s:
            for (f, (vi,im)) in enumerate(zip(detect(track(vi, stride=3), mirror=False, trackconf=0.2, minprob=minconf, maxdets=105, avgdets=70, throttle=True, activityiou=0.1),  # activity detection 
                                              vi.stream().frame(n=-detect.temporal_support()) if s is not None else itertools.repeat(None))):  # streaming visualization (n=delay)
                if s is not None:
                    s.write(self._annotator(im).rgb())
                print('%s, frame=%d' % (str(vi), f))                    
                
        vi.activityfilter(lambda a: a.category() not in ['person', 'person_walks', 'vehicle', 'car_moves'])   # remove background activities
        vo = vi.framerate(srcfps)  # upsample tracks/activities back to source framerate
        vo = vo.mindim(srcdim)  # upscale tracks back to source resolution
        gc.enable()

        return vo


    def annotate(self, v, outfile, minconf=0.1, trackonly=False, nounonly=False):
        return (v.activityfilter(lambda a: a.confidence() >= float(minconf))
                .annotate(mutator=vipy.image.mutator_show_trackindex_verbonly(confidence=True) if (not trackonly and not nounonly) else (vipy.image.mutator_show_trackonly() if trackonly else vipy.image.mutator_show_nounonly(nocaption=True)),
                          timestamp=True,
                          fontsize=6,
                          outfile=outfile))  # colored boxes by track id, activity captions with confidence, 5Hz, 512x(-1) resolution    
    
