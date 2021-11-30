import os
import sys
import torch
import vipy
import shutil
import numpy as np
import warnings
from vipy.util import remkdir, filetail, readlist, tolist, filepath, chunklistbysize, Timer
from heyvi.model.yolov3.network import Darknet
from heyvi.model.face.detection import FaceRCNN 
import copy
import heyvi.model.yolov5.models.yolo


class TorchNet(object):

    def gpu(self, idlist, batchsize=None):
        assert batchsize is None or (isinstance(batchsize, int) and batchsize > 0), "Batchsize must be integer"
        assert idlist is None or isinstance(idlist, int) or (isinstance(idlist, list) and len(idlist)>0), "Input must be a non-empty list of integer GPU ids"
        self._batchsize = int(batchsize if batchsize is not None else (self._batchsize if hasattr(self, '_batchsize') else 1))

        idlist = tolist(idlist)
        self._devices = ['cuda:%d' % k if k is not None and torch.cuda.is_available() and k != 'cpu' else 'cpu' for k in idlist]
        #self._tensortype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor       
        self._tensortype = torch.FloatTensor       

        if not hasattr(self, '_gpulist') or not hasattr(self, '_models') or idlist != self._gpulist  or not hasattr(self, '_models'):        
            self._models = [copy.deepcopy(self._model).to(d, non_blocking=False) for d in self._devices]
            for (d,m) in zip(self._devices, self._models):
                m.eval()
            self._gpulist = idlist
        torch.set_grad_enabled(False)
        return self

    def cpu(self, batchsize=None):
        return self.gpu(idlist=['cpu'], batchsize=batchsize)
    
    def iscpu(self):
        return any(['cpu' in d for d in self._devices])

    def isgpu(self):
        return any(['cuda' in d for d in self._devices])
    
    def __call__(self, t):
        """Parallel evaluation of tensor to split across GPUs set up in gpu().  t should be of size (ngpu*batchsize)
        
           * Note: Do not use DataParallel, this replicates the multi-gpu batch on device 0 and results in out of memory
        """
        assert len(t) <= self.batchsize(), "Invalid batch size"
        todevice = [b.pin_memory().to(d, non_blocking=True) for (b,d) in zip(t.split(self._batchsize) , self._devices)]  # async?
        fromdevice = [m(b) for (m,b) in zip(self._models, todevice)]   # async?
        return torch.cat([r.detach().cpu() for r in fromdevice], dim=0)
        
    def batchsize(self):
        return int(len(self._models)*self._batchsize)
        

class FaceDetector(TorchNet):
    """Faster R-CNN based face detector
    
    """

    def __init__(self, weightfile=None, gpu=None):    
        indir = os.path.join(filepath(os.path.abspath(__file__)), 'model', 'face')


        weightfile = vipy.downloader.downloadif('https://dl.dropboxusercontent.com/s/rdfre0oc456t5ee/resnet-101_faster_rcnn_ohem_iter_20000.pth',
                                                vipy.util.tocache('resnet-101_faster_rcnn_ohem_iter_20000.pth'),  # set VIPY_CACHE env
                                                sha1='a759030540a4a5284baa93d3ef5e47ed40cae6d6') if weightfile is None else weightfile
        
        self._model = FaceRCNN(model_path=weightfile)
        #self._model.eval()  # Set in evaluation mode

        #if gpu is not None:
        #    self.gpu(gpu, batchsize)
        #else:
        #    self.cpu()
        
    def __call__(self, im):
        assert isinstance(im, vipy.image.Image)
        return vipy.image.Scene(array=im.numpy(), colorspace=im.colorspace(), objects=[vipy.object.Detection('face', xmin=bb[0], ymin=bb[1], width=bb[2], height=bb[3], confidence=bb[4]) for bb in self._model(im)]).union(im)

    def batchsize(self):
        return 1  # FIXME

    
class Yolov5(TorchNet):
    """Yolov5 based object detector

       >>> d = heyvi.detection.Detector()
       >>> d(vipy.image.vehicles()).show()

    """
    
    def __init__(self, batchsize=1, weightfile=None, gpu=None):    
        self._mindim = 640  # must be square
        indir = os.path.join(filepath(os.path.abspath(__file__)), 'model', 'yolov5')
        cfgfile = os.path.join(indir, 'models', 'yolov5x.yaml')        
        weightfile = vipy.downloader.downloadif('https://dl.dropboxusercontent.com/s/jcwvz9ncjwpoat0/yolov5x.weights',
                                                vipy.util.tocache('yolov5x.weights'),  # set VIPY_CACHE env 
                                                sha1='bdf2f9e0ac7b4d1cee5671f794f289e636c8d7d4') if weightfile is None else weightfile

        # First import: load yolov5x.pt, disable fuse() in attempt_load(), save state_dict weights and load into newly pathed model
        with torch.no_grad():
            self._model = heyvi.model.yolov5.models.yolo.Model(cfgfile, 3, 80)
            self._model.load_state_dict(torch.load(weightfile))
            self._model.fuse()
            self._model.eval()

        self._models = [self._model]
        
        self._batchsize = batchsize        
        assert isinstance(self._batchsize, int), "Batchsize must be integer"
        self._cls2index = {c:k for (k,c) in enumerate(readlist(os.path.join(indir, 'coco.names')))}
        self._index2cls = {k:c for (c,k) in self._cls2index.items()}

        self._device = None
        #self._gpulist = gpu  # will be set in self.gpu()
        if gpu is not None:
            self.gpu(gpu, batchsize)
        else:
            self.cpu()
        torch.set_grad_enabled(False)
        
    def __call__(self, imlist, conf=1E-3, iou=0.5, union=False, objects=None):
        """Run detection on an image list at specific mininum confidence and iou NMS

           - yolov5 likes to split people into upper torso and lower body when in unfamilar poses (e.g. sitting, crouching)

        """
        assert isinstance(imlist, vipy.image.Image) or (isinstance(imlist, list) and all([isinstance(i, vipy.image.Image) for i in imlist])), "Invalid input - must be vipy.image.Image object and not '%s'" % (str(type(imlist)))
        assert objects is None or (isinstance(objects, list) and all([(k[0] if isinstance(k, tuple) else k) in self._cls2index for k in objects])), "Objects must be a list of allowable categories"
        objects = {(k[0] if isinstance(k,tuple) else k):(k[1] if isinstance(k,tuple) else k) for k in objects} if isinstance(objects, list) else objects

        with torch.no_grad():
            imlist = tolist(imlist)
            imlistdets = []
            t = torch.cat([im.clone(shallow=True).maxsquare().mindim(self._mindim).gain(1.0/255.0).torch(order='NCHW') for im in imlist])  # triggers load
            if torch.cuda.is_available() and not self.iscpu():
                t = t.pin_memory()

            assert len(t) <= self.batchsize(), "Invalid batch size: %d > %d" % (len(t), self.batchsize())
            todevice = [b.to(d, memory_format=torch.contiguous_format, non_blocking=True) for (b,d) in zip(t.split(self._batchsize), self._devices)]  # contiguous_format required for torch-1.8.1
            fromdevice = [m(b)[0] for (m,b) in zip(self._models, todevice)]     # detection
        
            t_out = [torch.squeeze(t, dim=0) for d in fromdevice for t in torch.split(d, 1, 0)]   # unpack batch to list of detections per imag
            t_out = [torch.cat((t[:,0:5], torch.argmax(t[:,5:], dim=1, keepdim=True)), dim=1) for t in t_out]  # filter argmax on device 
            t_out = [t[t[:,4]>conf].cpu().detach().numpy() for t in t_out]  # filter conf on device (this must be last)

        k_valid_objects = set([self._cls2index[k] for k in objects.keys()]) if objects is not None else self._cls2index.values()        
        for (im, dets) in zip(imlist, t_out):
            if len(dets) > 0:
                k_det = np.argwhere((dets[:,4] > conf).flatten() & np.array([int(d) in k_valid_objects for d in dets[:,5]])).flatten().tolist()
                objectlist = [vipy.object.Detection(xcentroid=float(dets[k][0]),
                                                    ycentroid=float(dets[k][1]),
                                                    width=float(dets[k][2]),
                                                    height=float(dets[k][3]),
                                                    confidence=float(dets[k][4]),
                                                    category='%s' % self._index2cls[int(dets[k][5])],
                                                    id=True)
                              for k in k_det]
                                 
                scale = max(im.shape()) / float(self._mindim)  # to undo
                objectlist = [obj.rescale(scale) for obj in objectlist]
                objectlist = [obj.category(objects[obj.category()]) if objects is not None else obj for obj in objectlist]  # convert to target class before NMS
            else:
                objectlist = []

            imd = im.objects(objectlist) if not union else im.objects(objectlist + im.objects())
            if iou > 0:
                imd = imd.nms(conf, iou)  
            imlistdets.append(imd)  
            
        return imlistdets if self._batchsize > 1 else imlistdets[0]

    def classlist(self):
        return list(self._cls2index.keys())
    
    
class Yolov3(TorchNet):
    """Yolov3 based object detector

       >>> d = heyvi.detection.Detector()
       >>> d(vipy.image.vehicles()).show()

    """
    
    def __init__(self, batchsize=1, weightfile=None, gpu=None):    
        self._mindim = 416  # must be square
        indir = os.path.join(filepath(os.path.abspath(__file__)), 'model', 'yolov3')
        cfgfile = os.path.join(indir, 'yolov3.cfg')
        self._model = Darknet(cfgfile, img_size=self._mindim)

        weightfile = vipy.downloader.downloadif('https://dl.dropboxusercontent.com/s/ve9cpuozbxh601r/yolov3.weights',
                                                vipy.util.tocache('yolov3.weights'),  # set VIPY_CACHE env 
                                                sha1='520878f12e97cf820529daea502acca380f1cb8e') if weightfile is None else weightfile
        
        self._model.load_darknet_weights(weightfile)
        self._model.eval()  # Set in evaluation mode
        self._batchsize = batchsize        
        assert isinstance(self._batchsize, int), "Batchsize must be integer"
        self._cls2index = {c:k for (k,c) in enumerate(readlist(os.path.join(indir, 'coco.names')))}
        self._index2cls = {k:c for (c,k) in self._cls2index.items()}

        self._device = None
        self._gpulist = gpu
        self.gpu(gpu, batchsize)
        
    def __call__(self, im, conf=5E-1, iou=0.5, union=False, objects=None):
        assert isinstance(im, vipy.image.Image) or (isinstance(im, list) and all([isinstance(i, vipy.image.Image) for i in im])), "Invalid input - must be vipy.image.Image object and not '%s'" % (str(type(im)))
        assert objects is None or (isinstance(objects, list) and all([(k[0] if isinstance(k, tuple) else k) in self._cls2index for k in objects])), "Objects must be a list of allowable categories"
        objects = {(k[0] if isinstance(k,tuple) else k):(k[1] if isinstance(k,tuple) else k) for k in objects} if isinstance(objects, list) else objects

        imlist = tolist(im)
        imlistdets = []
        t = torch.cat([im.clone().maxsquare().mindim(self._mindim).gain(1.0/255.0).torch(order='NCHW') for im in imlist]).type(self._tensortype)  # triggers load
        t_out = super().__call__(t).detach().numpy()   # parallel multi-GPU evaluation, using TorchNet()
        for (im, dets) in zip(imlist, t_out):
            k_class = np.argmax(dets[:,5:], axis=1).flatten().tolist()
            k_det = np.argwhere((dets[:,4] > conf).flatten() & np.array([((objects is None) or (self._index2cls[k] in objects.keys())) for k in k_class])).flatten().tolist()
            objectlist = [vipy.object.Detection(xcentroid=float(dets[k][0]),
                                                ycentroid=float(dets[k][1]),
                                                width=float(dets[k][2]),
                                                height=float(dets[k][3]),
                                                confidence=float(dets[k][4]),
                                                category='%s' % self._index2cls[k_class[k]],
                                                id=True)
                          for k in k_det]
            
            scale = max(im.shape()) / float(self._mindim)  # to undo
            objectlist = [obj.rescale(scale) for obj in objectlist]
            objectlist = [obj.category(objects[obj.category()]) if objects is not None else obj for obj in objectlist]
            imd = im.clone().array(im.numpy()).objects(objectlist).nms(conf, iou)  # clone for shared attributese
            imlistdets.append(imd if not union else imd.union(im))
            
        return imlistdets if self._batchsize > 1 else imlistdets[0]

    def classlist(self):
        return list(self._cls2index.keys())
    

class ObjectDetector(Yolov5):
    """Default object detector"""
    pass


class MultiscaleObjectDetector(ObjectDetector):  
    """Given a list of images, break each one into a set of overlapping tiles, and ObjectDetector() on each, then recombining detections"""
    def __call__(self, imlist, conf=0.5, iou=0.5, maxarea=1.0, objects=None, overlapfrac=6, filterborder=True, cover=0.7):  
        (f, n) = (super().__call__, self._mindim)
        assert isinstance(imlist, vipy.image.Image) or isinstance(imlist, list) and all([isinstance(im, vipy.image.Image) for im in imlist]), "invalid input"
        imlist = tolist(imlist)
        scale = imlist[0].mindim() / n
        
        (imlist_multiscale, imlist_multiscale_flat, n_coarse, n_fine) = ([], [], [], [])
        for im in imlist:
            imcoarse = [im]

            # FIXME: generalize this parameterization
            if overlapfrac == 6:
                imfine = (im.tile(n, n, overlaprows=im.height()-n, overlapcols=(3*n-im.width())//2) if (im.mindim()>=n and im.mindim() == im.height()) else
                          (im.tile(n, n, overlapcols=im.width()-n, overlaprows=(3*n-im.height())//2) if im.mindim()>=n else []))  # 2x3 tile, assumes im.mindim() == (n+n/2)
                if len(imfine) != 6:
                    print('WARNING: len(imtile) = %d for overlapfrac = %d' % (len(imfine), overlapfrac))  # Sanity check                    
                    
            elif overlapfrac == 2:
                imfine = (im.tile(n, n, overlaprows=0, overlapcols=(2*n-im.width())//2) if (im.mindim()>=n and im.mindim() == im.height()) else
                          (im.tile(n, n, overlapcols=0, overlaprows=(2*n-im.height())//2) if im.mindim()>=n else []))  # 1x2 tile, assumes im.mindim() == (n)
                if len(imfine) != 2:
                    print('WARNING: len(imtile) = %d for overlapfrac = %d' % (len(imfine), overlapfrac))  # Sanity check
                    
            elif overlapfrac == 0:
                imfine = []
                
            else:
                raise
            # /FIXME
            
            n_coarse.append(len(imcoarse))
            n_fine.append(len(imfine))
            imlist_multiscale.append(imcoarse+imfine)
            imlist_multiscale_flat.extend(imcoarse + [imf.maxsquare(n) for imf in imfine])            

        imlistdet_multiscale_flat = [im for iml in chunklistbysize(imlist_multiscale_flat, self.batchsize()) for im in tolist(f(iml, conf=conf, iou=0, objects=objects))]
        
        imlistdet = []
        for (k, (iml, imb, nf, nc)) in enumerate(zip(imlist, imlist_multiscale, n_fine, n_coarse)):
            im_multiscale = imlistdet_multiscale_flat[0:nf+nc]; imlistdet_multiscale_flat = imlistdet_multiscale_flat[nf+nc:];
            imcoarsedet = im_multiscale[0].mindim(iml.mindim())
            imcoarsedet_imagebox = imcoarsedet.imagebox()
            if filterborder:
                imfinedet = [im.nms(conf, iou, cover=cover).objectfilter(lambda o: ((maxarea==1 or (o.area()<=maxarea*im.area())) and   # not too big relative to tile
                                                                                    ((o.isinterior(im.width(), im.height(), border=0.9) or  # not occluded by any tile boundary 
                                                                                      o.clone().dilatepx(0.1*im.width()+1).cover(im.imagebox()) == o.clone().dilatepx(0.1*im.width()+1).set_origin(im.attributes['tile']['crop']).cover(imcoarsedet_imagebox)))))  # or only occluded by image boundary
                             for im in im_multiscale[nc:]]
                imfinedet = [im.objectmap(lambda o: o.set_origin(im.attributes['tile']['crop'])) for im in imfinedet]  # shift objects only, equivalent to untile() but faster
                imcoarsedet = imcoarsedet.objects( imcoarsedet.objects() + [o for im in imfinedet for o in im.objects()])  # union
            else:
                imfinedet = iml.untile( im_multiscale[nc:] )
                imcoarsedet = imcoarsedet.union(imfinedet) if imfinedet is not None else imcoarsedet
            imlistdet.append(imcoarsedet.nms(conf, iou, cover=cover))

        return imlistdet[0] if len(imlistdet) == 1 else imlistdet

    
class VideoDetector(ObjectDetector):  
    """Iterate ObjectDetector() over each frame of video, yielding the detected frame"""
    def __call__(self, v, conf=0.5, iou=0.5):
        assert isinstance(v, vipy.video.Video), "Invalid input"        
        for im in v.stream():
            yield super().__call__(im, conf=conf, iou=iou)

                        
class MultiscaleVideoDetector(MultiscaleObjectDetector):
    def __call__(self, v, conf=0.5, iou=0.5):
        assert isinstance(v, vipy.video.Video), "Invalid input"
        for imf in v.stream():
            yield super().__call__(imf, conf, iou)


class VideoTracker(ObjectDetector):
    def __call__(self, v, minconf=0.001, miniou=0.6, maxhistory=5, smoothing=None, objects=None, trackconf=0.05):
        (f, n) = (super().__call__, self._mindim)
        assert isinstance(v, vipy.video.Video), "Invalid input"
        assert objects is None or all([o in self.classlist() for o in objects]), "Invalid object list"
        vc = v.clone().clear()  
        for (k, vb) in enumerate(vc.stream().batch(self.batchsize())):
            for (j, im) in enumerate(tolist(f(vb.framelist(), minconf, miniou, union=False, objects=objects))):
                yield vc.assign(k*self.batchsize()+j, im.clone().objects(), minconf=trackconf, maxhistory=maxhistory)  # in-place            

    def track(self, v, minconf=0.001, miniou=0.6, maxhistory=5, smoothing=None, objects=None, trackconf=0.05, verbose=False):
        """Batch tracking"""
        for (k,vt) in enumerate(self.__call__(v.clone(), minconf=minconf, miniou=miniou, maxhistory=maxhistory, smoothing=smoothing, objects=objects, trackconf=trackconf)):
            if verbose:
                print('[heyvi.detection.VideoTracker][%d]: %s' % (k, str(vt)))  
        return vt


class FaceTracker(FaceDetector):
    def __call__(self, v, minconf=0.001, miniou=0.6, maxhistory=5, smoothing=None, trackconf=0.05, rescore=None, gate=64):
        (f, f_rescore) = (super().__call__, rescore if (rescore is not None and callable(rescore)) else (lambda im,k: im))
        assert isinstance(v, vipy.video.Video), "Invalid input"
        vc = v.clone()  
        for (k, vb) in enumerate(vc.stream().batch(self.batchsize())):
            for (j, im) in enumerate([f(im) for im in vb.framelist()]):
                frameindex = k*self.batchsize()+j
                yield vc.assign(frameindex, f_rescore(im.clone(), frameindex).objects(), minconf=trackconf, maxhistory=maxhistory, gate=gate)  # in-place            

    def track(self, v, minconf=0.001, miniou=0.6, maxhistory=5, smoothing=None, objects=None, trackconf=0.05, verbose=False, gate=64):
        """Batch tracking"""
        for (k,vt) in enumerate(self.__call__(v.clone(), minconf=minconf, miniou=miniou, maxhistory=maxhistory, smoothing=smoothing, trackconf=trackconf, gate=gate)):
            if verbose:
                print('[heyvi.detection.FaceTracker][%d]: %s' % (k, str(vt)))  
        return vt
    
    
class MultiscaleVideoTracker(MultiscaleObjectDetector):
    """MultiscaleVideoTracker() class

    Args:

        minconf: [float]: The minimum confidence of an object detection to be considered for tracking
        miniou: [float]: The minimum IoU of an object detection with a track to be considered for assignment
        maxhistory: [int]:  The maximum frame history lookback for assignment of a detection with a broken track
        smoothing: [str]:  Unused
        objects: [list]:  The list of allowable objects for tracking as supported by `heyvi.detection.MultiscaleObjectDetector`.
        trackconf: [float]: The minimum confidence of an unassigned detection to spawn a new track
        verbose: [bool]:  Logging verbosity
        gpu: [list]: List of GPU indexes to use
        batchsize: [int]:  The GPU batchsize
        weightfile: [str]: The modelfile for the object detector
        overlapfrac: [int]: FIXME, this is a legacy parameter
        detbatchsize: [int]:  The detection batchsize per image
        gate: [int]:  The maximum distance in pixels around a detection to search for candidate tracks

    """


    def __init__(self, minconf=0.05, miniou=0.6, maxhistory=5, smoothing=None, objects=None, trackconf=0.2, verbose=False, gpu=None, batchsize=1, weightfile=None, overlapfrac=6, detbatchsize=None, gate=64):
        super().__init__(gpu=gpu, batchsize=batchsize, weightfile=weightfile)
        self._minconf = minconf
        self._miniou = miniou
        self._maxhistory = maxhistory
        self._smoothing = smoothing
        self._objects = objects
        self._trackconf = trackconf
        self._verbose = verbose
        self._maxarea = 1.0
        self._overlapfrac = overlapfrac
        self._detbatchsize = detbatchsize if detbatchsize is not None else self.batchsize()
        self._gate = gate

    def _track(self, vi, stride=1, continuous=False, buffered=True, rescore=None):
        """Yield vipy.video.Scene(), an incremental tracked result for each frame.
        
            Args:
                rescore: [callable]: Takes in a single frame with objects and a frame index, and rescores confidences.  Useful for rescoring detections prior to tracking using prior or out-of-band information.  
        """
        assert isinstance(vi, vipy.video.Video), "Invalid input"
        assert rescore is None or callable(rescore), "Invalid input"        

        (det, n, k) = (super().__call__, self._mindim, 0)
        rescore = (lambda x,k: x) if rescore is None else rescore
        for (k,vb) in enumerate(vi.stream(buffered=buffered).batch(self._detbatchsize)):
            framelist = vb.framelist()
            for (j, im) in zip(range(0, len(framelist), stride), tolist(det(framelist[::stride], self._minconf, self._miniou, self._maxarea, objects=self._objects, overlapfrac=self._overlapfrac))):
                for i in range(j, j+stride):                    
                    if i < len(framelist):
                        frameindex = k*self._detbatchsize+i
                        yield (vi.assign(frame=frameindex, dets=rescore(im, frameindex).objects(), minconf=self._trackconf, maxhistory=self._maxhistory, gate=self._gate) if (i == j) else vi)
                            
    def __call__(self, vi, stride=1, continuous=False, buffered=True):
        return self._track(vi, stride, continuous, buffered=buffered)
    
    def stream(self, vi):
        return self._track(vi)

    def track(self, vi, verbose=False):
        """Batch tracking"""
        for v in self.stream(vi):
            if verbose:
                print(vi)
        return vi
        

class WeakAnnotationTracker(MultiscaleVideoTracker):
    """heyvi.detection.WeakAnnotationTracker()

    Given a weak annotation of an object bounding box from a human annotator, refine this weak annotation into a tight box using object detection proposals and tracking.
    
    Approach:

        - The input video should have weak tracks provided by live annotators with class names that intersect `heyvi.detection.MultiscaleVideoTracker`.
        - Weak annotations are too loose, too tight, or poorly centered boxes provided by live annotators while recording.  
        - This function runs a low confidence object detector and rescores object detection confidences based on overlap with the proposal.  
        - Detections that maximally overlap the proposal with high detection confidence are proritized for tracking.
        - The tracker compbines these rescored detections as in the VideoTracker.
        - When done, each proposal is assigned to one track, and track IDs and activity IDs are mappped accordingly. 
        - Activities that no longer overlap the actor track are removed

    Usage:

    Batch annotation tracker:

    ```python
    T = heyvi.detection.WeakAnnotationTracker()
    v = vipy.video.Scene(...)  # contains weak annotations
    vt = T.track(v)   # refined proposals
    vm = vt.combine(v.trackmap(lambda t: t.category('weak annotation')))
    ```

    Streaming annotation tracker:

    ```python
    T = heyvi.detection.WeakAnnotationTracker()
    v = vipy.video.Scene(...)  # contains weak annotations
    for vt in T(v):
        print(vt)
    ```

    .. note::
        - The video vt will be a clone of v such that each track in vt will be a refined track of a track in v.  
        - All track and activities IDs are mapped appropriately from the input video.  
        - The combined video vm has both the weak annotation and the refined tracks.

    """
    def __init__(self, minconf=0.001, miniou=0.6, maxhistory=128, trackconf=0.005, verbose=False, gpu=None, batchsize=1, weightfile=None, overlapfrac=0, detbatchsize=None, gate=256):
        # Reduced default minimum confidence for detections and track confidence for spawning new tracks to encourage selection of best weak annotation box
        # Increased maxhistory with wide measurement assignment gate to reacquire lost tracks (detections are weighted by weak annotation alignment, so wide gate is low risk)
        super().__init__(minconf=minconf, miniou=miniou, maxhistory=maxhistory, objects=None, trackconf=trackconf, verbose=verbose, gpu=gpu, batchsize=batchsize, weightfile=weightfile, overlapfrac=overlapfrac, detbatchsize=detbatchsize, gate=gate)

    def _track(self, vi, stride=1, continuous=False, buffered=True):
        # Object rescoring: Detection confidence of each object is rescored by multiplying confidence by the max IoU (or max cover) with a weak object annotation of the same category
        f_rescorer = lambda im, f, va=vi.clone(): im.objectmap(lambda o, ima=va.frame(f, noimage=True): o.confidence(o.confidence()*max([1e-1]+[max(a.iou(o), a.cover(o)) for a in ima.objects() if a.category().lower() == o.category().lower()])))
        return super()._track(vi.clone().cleartracks(), stride=stride, continuous=continuous, buffered=buffered, rescore=f_rescorer)

    def track(self, vi, verbose=False):
        self._objects = list(set([t.category().lower() for t in vi.tracklist()]).intersection(set(self.classlist())))  # only detect weakly annotated objects
        vt = super().track(vi.clone(), verbose=verbose)
        if len(vt.tracks()) > 0:
            for ti in vi.tracklist():
                t = max(vt.tracklist(), key=lambda t: ti.iou(t)*t.confidence()*float(t.category().lower() == ti.category().lower()))  # best track for weak annotation
                vt.rekey( tracks={t.id():ti.id()}, activities={} )  # set assigned track ID for activity association, no change to activities
        return vt.trackfilter(lambda t: t.id() in vi.tracks()).activityfilter(lambda a: a.actorid() is not None and a.hastrackoverlap(vt.track(a.actorid())))
            

class WeakAnnotationFaceTracker(FaceTracker):    
    """heyvi.detection.WeakAnnotationFaceTracker()

    Given a weak annotation of an person, face or head bounding box from a human annotator, refine this weak annotation into a tight box around the face using object detection proposals and tracking.
    
    Approach:

        - The input video should have weak tracks provided by live annotators with class names that are in ['person', 'face', 'head']
        - Weak annotations are too loose, too tight, or poorly centered boxes provided by live annotators while recording.  
        - This function runs a low confidence face detector and rescores face detection confidences based on overlap with the proposal.  
        - Detections that maximally overlap the proposal with high detection confidence are proritized for track assignment.
        - The tracker compbines these rescored detections as in the VideoTracker.
        - When done, each track is assigned to a proposal. 

    See also: `heyvi.detection.WeakAnnotationTracker`
    """
    
    def __init__(self, minconf=0.001, miniou=0.6, maxhistory=128, trackconf=0.005, gpu=None, gate=256):
        # Reduced default minimum confidence for detections and track confidence for spawning new tracks to encourage selection of best weak annotation box
        super().__init__(gpu=gpu)
        self._minconf = minconf
        self._gate = gate
        self._trackconf = trackconf
        self._maxhistory = maxhistory
        self._miniou = miniou

    def __call__(self, vi, minconf, miniou, maxhistory, trackconf, gate, smoothing=None):
        # Object rescoring: Detection confidence of each object is rescored by multiplying confidence by the max IoU (or max cover) with a weak object annotation of the same category
        f_rescorer = lambda im, f, va=vi.clone(): im.objectmap(lambda o, ima=va.frame(f, noimage=True): o.confidence(o.confidence()*max([1e-1]+[max(a.iou(o), a.cover(o)) for a in ima.objects() if o.category().lower() in ['face','head'] and  a.category().lower() in ['face','head','person']])))
        return super().__call__(vi.clone().clear(), minconf=self._minconf, miniou=self._miniou, maxhistory=self._maxhistory, smoothing=None, trackconf=self._trackconf, rescore=f_rescorer, gate=self._gate)

    def track(self, vi, verbose=False):        
        assert isinstance(vi, vipy.video.Scene)
        if not any([t.category().lower() in ['face','head', 'person'] for t in vi.tracklist()]):
            warnings.warn('No face proposals')
            return vi.clone()

        vic = vi.clone().trackfilter(lambda t: t.category().lower() in ['face','head', 'person']).clearactivities()
        vt = super().track(vic, verbose=verbose, maxhistory=self._maxhistory, minconf=self._minconf, trackconf=self._trackconf)
        if len(vt.tracks()) > 0:
            for ti in vic.tracklist():
                t = max(vt.tracklist(), key=lambda t: max(ti.iou(t), ti.segmentcover(t))*t.confidence())  # best track for weak annotation
                vt.rekey( tracks={t.id():ti.id()}, activities={} )  # set assigned track ID for activity association, no change to activities
        return vt.trackfilter(lambda t: t.id() in vic.tracks())  
    

class ActorAssociation(MultiscaleVideoTracker):
    """heyvi.detection.VideoAssociation() class
       
       Select the best object track of the target class associated with the primary actor class by gated spatial IOU and distance.
       Add the best object track to the scene and associate with all activities performed by the primary actor.

    .. warning:: This is scheduled for deprecation, as the gating is unreliable.  This should be replaced by the WeakAnnotationTracker for a target class. 
    """

    @staticmethod
    def isallowable(v, actor_class, association_class, fps=None):
        allowable_objects = ['person', 'vehicle', 'car', 'motorcycle', 'object', 'bicycle', 'motorbike', 'truck']        
        return (actor_class.lower() in allowable_objects and
                all([a.lower() in allowable_objects for a in vipy.util.tolist(association_class)]) and
                actor_class.lower() in v.objectlabels(lower=True))
        

    def __call__(self, v, actor_class, association_class, fps=None, dilate=2.0, activity_class=None, maxcover=0.8, max_associations=1, min_confidence=0.4):
        allowable_objects = ['person', 'vehicle', 'car', 'motorcycle', 'object', 'bicycle', 'motorbike', 'truck']        
        association_class = [a.lower() for a in vipy.util.tolist(association_class)]
        assert actor_class.lower() in allowable_objects, "Primary Actor '%s' not in allowable target class '%s'" % (actor_class.lower(), str(allowable_objects))
        assert all([a in allowable_objects for a in allowable_objects]), "Actor Association '%s' not in allowable target class '%s'" % (str(association_class), str(allowable_objects))
        assert actor_class.lower() in v.objectlabels(lower=True), "Actor Association can only be performed with scenes containing an allowable actor not '%s'" % str(v.objectlabels())
        
        # Track objects
        vc = v.clone()
        if fps is not None:
            for t in vc.tracks().values():
                t._framerate = v.framerate()  # HACK: backwards compatibility
            for a in vc.activities().values():
                a._framerate = v.framerate()  # HACK: backwards compatibility
        vc = vc.framerate(fps) if fps is not None else vc   # downsample
        vt = self.track(vc.clone())  # track at downsampled framerate

        # Actor assignment: for every activity, find track with best target object assignment to actor (first track in video)
        for a in vc.activities().values():
            candidates = [t for t in vt.tracks().values() if (t.category().lower() in association_class and
                                                              t.during_interval(a.startframe(), a.endframe()) and
                                                              t.confidence() > min_confidence and  # must have minimum confidence
                                                              (actor_class.lower() not in association_class or t.segmentcover(vc.actor()) < maxcover) and
                                                              vc.actor().boundingbox().dilate(dilate).hasintersection(t.boundingbox()))] # candidate assignment (cannot be actor, or too far from actor)
            if len(candidates) > 0:
                # best assignment is track closest to actor with maximum confidence and minimum dilated overlap
                trackconf = sorted([(t, vc.actor().boundingbox().dilate(dilate).iou(t.boundingbox()) * t.confidence()) for t in candidates], key=lambda x: x[1], reverse=True)
                for (t, conf) in trackconf[0:max_associations]:
                    if a.during_interval(t.startframe(), t.endframe()) and activity_class is None or a.category() == activity_class:
                        a.add(t)
                        vc.add(t)

        return vc.framerate(v.framerate()) if vc.framerate() != v.framerate() else vc   # upsample

    
