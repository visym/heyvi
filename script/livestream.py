import os
import vipy
from argparse import ArgumentParser


def videostream():
    assert 'VIPY_RTSP_KEY' in os.environ and 'VIPY_YOUTUBE_STREAM_KEY' in os.environ
    
    vi = vipy.video.Scene(url='rtsp://%s@10.0.1.14/live0' % os.environ['VIPY_RTSP_KEY'])
    vo = vipy.video.Scene(url='rtmp://a.rtmp.youtube.com/live2/%s' % os.environ['VIPY_YOUTUBE_STREAM_KEY'])
    
    with vo.stream(write=True) as s:
        for (k,im) in enumerate(vi.stream()):
            print(k,im)
            s.write(im)

            
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-v","--video", help="Livestream video only", action='store_true')
    args = parser.parse_args()

    if args.video:
        videostream()
