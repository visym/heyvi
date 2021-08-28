import os
import vipy

import os
import gc

import torch
import heyvi.recognition
import heyvi.detection
import pycollector.version
import pycollector.label
import heyvi.version



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
    """Record a livestream to an output video file"""
    def __init__(self, outfile, fps=30):
        assert vipy.util.isvideo(outfile)
        self._vo = vipy.video.Scene(filename=outfile, framerate=fps)

    def __repr__(self):
        return '<heyvi.system.Recorder: %s>' % str(self._vo)
    
    def __call__(self, vi):
        assert isinstance(vi, vipy.video.Scene)
        
        with self._vo.stream(overwrite=True) as s:
            for (k,im) in enumerate(vi.stream()):
                print(k,im)
                s.write(im)


class Actev21():
    def __init__(self):

        assert vipy.version.is_exactly('1.11.7')
        assert heyvi.version.is_exactly('0.0.3')
        #assert torch.cuda.device_count() >= 4
        
        self._activitymodel = vipy.downloader.downloadif('https://dl.dropboxusercontent.com/s/ntvjg352b0fwnah/mlfl_v5_epoch_41-step_59279.ckpt',
                                                         vipy.util.tocache('mlfl_v5_epoch_41-step_59279.ckpt'),  # set VIPY_CACHE env 
                                                         sha1='c4457e5b2e4fa1462d552070c47cac9eb2833e47')

    def __call__(self, vi):

        assert isinstance(vi, vipy.video.Video)
        
        objects = ['person', ('car','vehicle'), ('truck','vehicle'), ('bus', 'vehicle'), 'bicycle']  # merge truck/bus/car to vehicle, no motorcycles
        track = heyvi.detection.MultiscaleVideoTracker(gpu=[0,1,2,3], batchsize=9, minconf=0.05, trackconf=0.2, maxhistory=5, objects=objects, overlapfrac=6, gate=64, detbatchsize=None)
        activities = list(pycollector.label.pip_plus_meva_to_meva.items())
        detect = heyvi.recognition.ActivityTracker(gpus=[0,1,2,3], batchsize=64, modelfile=self._activitymodel, stride=3, activities=activities)   # stride should match tracker stride 4->3

        gc.disable()
        (srcdim, srcfps) = (vi.mindim(), vi.framerate())
        vi = vi.mindim(960).framerate(5)        
        with vipy.util.Stopwatch() as t:
            for (f,vi) in enumerate(detect(track(vi, stride=3), mirror=False, trackconf=0.2, minprob=0.04, maxdets=105, avgdets=70, throttle=True, activityiou=0.1)):   
                print('%s, frame=%d' % (str(vi), f))
                
        vi.activityfilter(lambda a: a.category() not in ['person', 'person_walks', 'vehicle', 'car_moves'])   # remove background activities
        vi.activityfilter(lambda a: strict is False or a.category() in activitylist)  # remove invalid activities (if provided)
        vo = vi.framerate(srcfps)  # upsample tracks/activities back to source framerate
        vo = vo.rescale(srcdim / 960.0)  # upscale tracks back to source resolution
        gc.enable()
        
        return vo
    
