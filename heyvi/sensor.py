import os
import vipy


def rtsp(url=None, fps=30):
    """Return an RTSP camera.

    >>> v = heyvi.sensor.rtsp()
    >>> im = v.preview().show().saveas('out.jpg')
    >>> for im in v:
    >>>     print(im)  # live stream 
    >>>     print(im.numpy())  # of numpy frames

    Args:
        url: [str]  The URL for the rtsp camera, must start with 'rtsp://'
        fps: [float]  The framerate of the returned camera, can also be set after
    
    Env:
        VIPY_RTSP_URL:  If this environment variable is set, use this as the URL that contains integrated credentials
    
    """
    
    assert url is not None or 'VIPY_RTSP_URL' in os.environ
    url = url if url is not None else os.environ['VIPY_RTSP_URL']
    assert vipy.util.isRTSPurl(url)
    return vipy.video.Scene(url=url, framerate=fps)    



