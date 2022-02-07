import vipy
import heyvi
from argparse import ArgumentParser
import heyvi.cap


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i","--infile", help="Input video file (.mp4) to track", required=True)
    parser.add_argument("-f","--framerate", help="Input video file framerate for tracking", default=5)
    parser.add_argument("-j","--jsonfile", help="Output JSON with tracked annotations")    
    parser.add_argument("-o","--outfile", help="Output video file with tracked annotations")
    parser.add_argument("-c","--confidence", help="Minimum activity confidence for visualization", default=0.15)
    args = parser.parse_args()

    T = heyvi.cap.CAP()
    v = T.detect(vipy.video.Scene(filename=args.infile).framerate(args.framerate), minconf=float(args.confidence))

    if args.outfile:
        T.annotate(v, outfile=outfile, minconf=minconf)
        
    if args.jsonfile:
        vipy.util.save(v, args.jsonfile)



    
