import os
import vipy
from argparse import ArgumentParser
import heyvi.util


def youtube(vi=None):
    """Stream video camera to youtube live"""
    assert (vi is None and 'VIPY_RTSP_URL' in os.environ) or isinstance(vi, vipy.video.Video)

    vi = vipy.video.Scene(url=os.environ['VIPY_RTSP_URL'], framerate=30).mindim(512) if vi is None else vi
    return heyvi.system.YouteubLive(vi)
    

            
def videofile(outfile, vi=None):
    """Live stream video camera to video file"""
    assert (vi is None and 'VIPY_RTSP_URL' in os.environ) or isinstance(vi, vipy.video.Video)    
    
    vi = vipy.video.Scene(url=os.environ['VIPY_RTSP_URL']) if vi is None else vi
    vo = vipy.video.Scene(filename=outfile)
    
    with vo.stream(overwrite=True) as s:
        for (k,im) in enumerate(vi.stream()):
            s.write(im.annotate(timestamp=heyvi.util.timestamp()).rgb())
            print(k,im)            

            
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-y","--youtube", help="Livestream video only to youtube", action='store_true')
    parser.add_argument("-v","--video", help="Livestream to an output video file")
    args = parser.parse_args()

    if args.video is not None:
        videofile(args.video)
    
    elif args.youtube:
        youtube()

        
