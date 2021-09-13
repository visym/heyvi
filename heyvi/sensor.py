import os
import vipy


def rtsp(url=None, fps=30):
    assert url is not None or 'VIPY_RTSP_URL' in os.environ
    url = url if url is not None else os.environ['VIPY_RTSP_URL']
    assert vipy.util.isRTSPurl(url)
    return vipy.video.Scene(url=url, framerate=fps)    



