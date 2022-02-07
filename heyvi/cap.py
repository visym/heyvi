import vipy
import heyvi
import torch
import pycollector
import pycollector.version
import pycollector.label
import contextlib
import gc




class CAP():
    """heyvi.system.CAP class

    """
    
    def __init__(self, modelfile=None):
        
        assert vipy.version.is_at_least('1.12.4')
        assert heyvi.version.is_at_least('0.2.13')
        assert pycollector.version.is_at_least('0.4.2')        
        assert torch.cuda.device_count() >= 4
        self._unitnorm = False
        
        #self._activitymodel = './cap_epoch_15_step_64063.ckpt'  # local testing only
        #self._activitymodel = './cap_epoch_17_step_72071.ckpt'  # local testing only
        #self._activitymodel = './_calibrate.ckpt'  # local testing only
        self._activitymodel = './cap_l2norm_e23s96095.ckpt' if modelfile is None else modelfile  # local testing only        
        self._unitnorm = True

        self._annotator = lambda im, f=vipy.image.mutator_show_trackindex_verbonly(confidence=True): f(im).annotate(timestamp=heyvi.util.timestamp(), timestampoffset=(6,10), fontsize=15).rgb()        

        
    def __call__(self, vi, minconf=0.04, verbose=True, frame_callback=None, livestream=False, mintracklen=None, finalized=True):

        assert isinstance(vi, vipy.video.Scene)

        livedelay = 2*15*5 if vi.islive() or livestream else 5 
        objects = ['person', ('car','vehicle'), ('truck','vehicle'), ('bus', 'vehicle'), 'bicycle']  # merge truck/bus/car to vehicle, no motorcycles
        track = heyvi.detection.MultiscaleVideoTracker(gpu=[0,1,2,3], batchsize=9, minconf=0.05, trackconf=0.2, maxhistory=5, objects=objects, overlapfrac=6, gate=64, detbatchsize=None)
        detect = heyvi.recognition.ActivityTrackerCap(gpus=[0,1,2,3], batchsize=64, modelfile=self._activitymodel, stride=3, unitnorm=self._unitnorm)   # stride should match tracker stride 4->3
        
        gc.disable()
        (srcdim, srcfps) = (vi.mindim(), vi.framerate())
        vi = vi.mindim(960).framerate(5)
        for (f, (im,vi)) in enumerate(zip(vi.stream(buffered=True).frame(delay=livedelay),  # live stream delay (must be >= 2x finalized period)
                                          detect(track(vi, stride=3, buffered=vi.islive()),
                                                 mirror=False, trackconf=0.2, minprob=minconf, maxdets=105, avgdets=70, throttle=True, activityiou=0.1, buffered=vi.islive(), finalized=(livedelay//2) if vi.islive() or livestream else finalized, mintracklen=mintracklen))):
            if callable(frame_callback) and im is not None:
                frame_callback(self._annotator(im.clone()), im, vi)  
            if verbose:
                print('[heyvi.system.Actev21][%s][%d]: %s' % (heyvi.util.timestamp(), f, vi), end='\r')                                    
                
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
    
    
    def detect(self, vi, minconf=0.15):
        assert isinstance(vi, vipy.video.Scene)
        return self.__call__(vi.clone().clear().framerate(5), minconf=minconf)

    
    def classify(self, vi, minconf=0.01, topk=3, repeat=3):
        assert isinstance(vi, vipy.video.Scene)
        v = vi.clone().clear().framerate(5).load()
        v = v.fromframes([vj for k in range(repeat) for vj in v.framelist()], copy=True)  # repeat to achieve minimums
        v = self.__call__(v, minconf=minconf, finalized=False)
        ai = set([a.id() for a in sorted(v.activitylist(), key=lambda a: a.confidence())[-topk:]])
        return v.activityfilter(lambda a: a.id() in ai).activities({a.id():a for a in sorted(v.activities().values(), key=lambda a: a.confidence(), reverse=True)})
