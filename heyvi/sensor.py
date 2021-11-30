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


def camera(n=None):
    """Return RSTP camera with index n in cameralist, or the default RTSP camera if None"""
    cameras = cameralist(online=False)
    assert n is None or (n >= 0 and n < len(cameras))
    return cameras[n] if n is not None else rtsp()


def cameralist(online=False):
    """Return all online RTSP cameras set up on the current network.

    This requires setting environment variables:

    VIPY_RTSP_URL_0='rtsp://user:passwd@ip.addr.0'
    VIPY_RTSP_URL_1='rtsp://user:passwd@ip.addr.1'
    VIPY_RTSP_URL_2='rtsp://user:passwd@ip.addr.2'

    Args:

        online: [bool]: If True, return only those cameras that are online.  If a camera is offline return None in that camera index.  If false, return all cameras specified by the environment variables.

    """
    return [rtsp(url=os.environ[k]) if (not online or rtsp(url=os.environ[k]).canload()) else None
            for k in sorted([k for k in os.environ.keys() if k.startswith('VIPY_RTSP_URL_')], key=lambda x: int(x.split('_')[-1]))]
