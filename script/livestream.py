import os
import vipy
from argparse import ArgumentParser
import heyvi.util

def youtube_livestream():
    assert 'VIPY_RTSP_KEY' in os.environ and 'VIPY_YOUTUBE_STREAM_KEY' in os.environ
    
    vi = vipy.video.Scene(url='rtsp://%s@10.0.1.14/live0' % os.environ['VIPY_RTSP_KEY'])
    vo = vipy.video.Scene(url='rtmp://a.rtmp.youtube.com/live2/%s' % os.environ['VIPY_YOUTUBE_STREAM_KEY'])
    
    with vo.stream(write=True) as s:
        for (k,im) in enumerate(vi.stream()):
            print(k,im)
            s.write(im)

            
def video_livestream(outfile):
    assert 'VIPY_RTSP_KEY' in os.environ 
    
    vi = vipy.video.Scene(url='rtsp://%s@10.0.1.14/live0' % os.environ['VIPY_RTSP_KEY'])
    vo = vipy.video.Scene(filename=outfile)
    
    with vo.stream(overwrite=True) as s:
        for (k,im) in enumerate(vi.stream()):
            s.write(im.annotate(timestamp=heyvi.util.timestamp()).rgb())
            print(k,im)            
            
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-y","--youtube", help="Livestream video only to yourtube", action='store_true')
    parser.add_argument("-v","--video", help="Livestream to an output video file")
    args = parser.parse_args()

    if args.video is not None:
        video_livestream(args.video)
    
    elif args.youtube:
        youtube_liveostream()

        
