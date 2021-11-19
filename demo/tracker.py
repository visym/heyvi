import vipy
import heyvi
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i","--infile", help="Input video file (.mp4) to track", required=True)
    parser.add_argument("-f","--framerate", help="Input video file framerate for tracking", default=5)    
    parser.add_argument("-o","--outfile", help="Output video file with tracked annotations", default='tracker.mp4')
    args = parser.parse_args()

    T = heyvi.system.Tracker()
    vi = vipy.video.Scene(filename=args.infile).framerate(args.framerate)
    vo = T(vi).annotate(args.outfile)
    print(vo)

    
