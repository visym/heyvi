import os
from argparse import ArgumentParser
import gc


def process_chunk(videofilelist, activitylist, frameratelist, cliplist, strict=False, outvideo=False, do_spatial_localization=False, dt_spatial_localization=10):

    import vipy  # installed via diva_evaluation_cli/src/implementation/setup.sh
    import torch
    import pycollector.recognition
    import pycollector.detection
    import pycollector.version
    import pycollector.label
        
    assert vipy.version.is_exactly('1.11.5')
    assert pycollector.version.is_exactly('0.2.8')
    
    modeldir = vipy.util.filepath(os.path.abspath(__file__))
    trackmodel = os.path.join(modeldir, 'yolov5x.weights')
    activitymodel = os.path.join(modeldir, 'mlfl_v5_epoch_41-step_59279.ckpt')
    #activitymodel = os.path.join(modeldir, 'mlfl_v6_epoch_24-step_36083.ckpt')
    #activitymodel = os.path.join(modeldir, 'mlfl_v7_epoch_29-step_38675.ckpt')    
    #activitymodel = os.path.join(modeldir, 'mlfl_v8_mevaonly_epoch_30-step_4823.ckpt')    
    assert torch.cuda.device_count() >= 4
    
    objects = ['person', ('car','vehicle'), ('truck','vehicle'), ('bus', 'vehicle'), 'bicycle']  # merge truck/bus/car to vehicle, no motorcycles
    track = pycollector.detection.MultiscaleVideoTracker(gpu=[0,1,2,3], batchsize=9, weightfile=trackmodel, minconf=0.05, trackconf=0.2, maxhistory=5, objects=objects, overlapfrac=6, gate=64, detbatchsize=None)
    activities = list(pycollector.label.pip_plus_meva_to_meva.items())
    detect = pycollector.recognition.ActivityTracker(gpus=[0,1,2,3], batchsize=64, modelfile=activitymodel, stride=3, activities=activities)   # stride should match tracker stride 4->3

    jsonlist = []
    outvideolist = []
    gc.disable()                        
    for (videofile, framerate, (startframe, endframe)) in zip(videofilelist, frameratelist, cliplist):        
        assert os.path.exists(videofile), "Invalid path '%s'" % videofile
        if not all([a in pycollector.label.pip_to_meva.values() for a in activitylist]):
            print('WARNING: requested activity list "%s" does not match source network labels "%s"' % (str(activitylist), str(list(pycollector.label.pip_to_meva.values()))))
            print('WARNING: disabling filtering requested activities')            
            strict = False  # disable me
        
        # HACK: See email trail with Jonathan F. and Andrew D. on 06JAN21 to avoid ffmpeg preview error on some SDL videos by setting startframe=0, not startframe=1 and avoiding preview
        if startframe == 1:
            print('WARNING: setting startframe=%s -> startframe=0 to avoid FFMPEG preview error on some SDL videos' % str(startframe))
            startframe = 0
        vi = vipy.video.Scene(filename=videofile, framerate=framerate).shape(probe=True).clip(startframe, endframe).mindim(960).framerate(5)        
        with vipy.util.Stopwatch() as t:
            for (f,vi) in enumerate(detect(track(vi, stride=3), mirror=False, trackconf=0.2, minprob=0.04, maxdets=105, avgdets=70, throttle=True, activityiou=0.1)):   
                print('%s, frame=%d' % (str(vi), f))
                
        vi.activityfilter(lambda a: a.category() not in ['person', 'person_walks', 'vehicle', 'car_moves'])   # remove background activities
        vi.activityfilter(lambda a: strict is False or a.category() in activitylist)  # remove invalid activities (if provided)
        vo = vi.framerate(framerate)  # upsample tracks/activities back to source framerate
        vo = vo.rescale(vipy.video.Scene(filename=videofile).mindim() / 960.0)  # upscale tracks back to source resolution

        if outvideo:
            print('Tracks = %s' % (str(vo.tracklist())))            
            print('Activities = %s' % (str(vo.activitylist())))
        print('Relative processing time = %1.3f\n' % float(t.duration() / ((endframe-startframe)/float(framerate))))

        filename = vipy.util.filetail(vo.filename())
        d = {'filesProcessed':[filename],
             'processingReport':[{'status':'success', 'message':''}],
             'activities':[{'activity':str(a.category()),
                            'activityID':int(j),  # must be integer
                            'presenceConf':float(a.confidence()),
                            'alertFrame':int(a.endframe()),
                            'localization':{filename: {str(a.startframe()):1,  # activity localization will always be on track boundaries
                                                       str(a.endframe()):0}},                            
                            'objects':[{'objectType':t.category(),
                                        'objectID':int(oi),  # int(str(ti), 16) results in overflow
                                        'localization':{filename:{str(kf):{'boundingBox':{"x":int(t[kf].xmin()), "y":int(t[kf].ymin()), "w":int(t[kf].width()), "h":int(t[kf].height())}}
                                                                  for kf in sorted(set([a.startframe(), a.endframe()] + t.keyframes())) if a.during(kf) and t[kf] is not None}}}
                                       for (oi, (ti,t)) in enumerate(vo.tracks().items()) if a.actorid() == ti] if do_spatial_localization else []}
                            for (j,(k,a)) in enumerate(vo.activities().items()) if not do_spatial_localization or vo.track(a.actorid()).during_interval(a.startframe(), a.endframe())]}
        
        if do_spatial_localization:
            # Filter invalid JSON, see email from Baptiste 06May21            
            d['activities'] = [a for a in d['activities'] if len(a['objects']) > 0 and all([len(kfkb) > 0 for o in a['objects'] for (f,kfkb) in o['localization'].items()])]  
        
        jsonlist.append(d)        
        outvideolist.append(vo.json() if outvideo else None)
        del vi, vo; gc.collect()
        
    gc.enable()                    
    return (jsonlist, outvideolist)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--outfile", help="Output visualization video file")
    parser.add_argument("--infile", help="Input video file", required=True)
    parser.add_argument("--framerate", help="Input video framerate", type=float, default=30.0)    
    parser.add_argument("--startframe", help="Input video startframe", type=int, required=True)
    parser.add_argument("--endframe", help="Input video endframe", type=int, required=True)
    parser.add_argument("--minconf", help="Minimum activity detection confidence for display [0,1]", type=float, default=0.5)
    parser.add_argument("--outjson", help="Output video json")
    parser.add_argument("--trackonly", help="Visualize tracks only", action='store_true')
    parser.add_argument("--nounonly", help="Visualize tracks only", action='store_true')    
    args = parser.parse_args()
    
    import vipy
    assert args.outjson is None or vipy.util.isjsonfile(args.outjson), "invalid input"
    assert args.minconf >= 0 and args.minconf <= 1, "invalid input"
    
    # Generate a visualization video for all activity detections in this video
    #   >>> python system.py --outfile=/path/to/overlay.mp4 --infile=/path/to/source.mp4 --framerate=30.0 --startframe=0 --endframe=1000
    #   >>> python system.py --infile=./actev-data-repo/corpora/VIRAT-V1/0400/VIRAT_S_040003_02_000197_000552.mp4 --outfile=/tmp/out.mp4 --minconf=0.5 --startframe=0 --endframe=9000 --outjson=/tmp/out.json
    #   >>> python system.py --infile=./actev-data-repo/corpora/VIRAT-V1/0000/VIRAT_S_000007.mp4 --outfile=/tmp/out.mp4 --minconf=0.5 --startframe=0 --endframe=9000 --outjson=/tmp/out.json 
    (d,V) = process_chunk([args.infile], [], [args.framerate], [(args.startframe, args.endframe)], strict=False, outvideo=True)    
    v = vipy.video.Scene.from_json(V[0])
    if args.outjson:
        vipy.util.save(v, args.outjson)

    if args.outfile:
        print(v.mindim(512)
              .activityfilter(lambda a: a.confidence() >= float(args.minconf))
              .annotate(mutator=vipy.image.mutator_show_trackindex_verbonly(confidence=True) if (not args.trackonly and not args.nounonly) else (vipy.image.mutator_show_trackonly() if args.trackonly else vipy.image.mutator_show_nounonly(nocaption=True)),
                        timestamp=True,
                        fontsize=6,
                        outfile=args.outfile))  # colored boxes by track id, activity captions with confidence, 5Hz, 512x(-1) resolution    

    
