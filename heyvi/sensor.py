import os
import vipy


def officecam(fps=30):
    return vipy.video.Scene(url='rtsp://%s@10.0.1.14/live0' % os.environ['VIPY_RTSP_KEY'], framerate=fps)
