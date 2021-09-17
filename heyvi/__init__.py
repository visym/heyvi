"""

# \"Hey Vi!\"

HEYVI is a python package for visual AI that provides systems and trained models for activity detection and object tracking in videos.

HEYVI provides:  

* Real time activity detection of the [MEVA activity classes](https://mevadata.org)
* Real time visual object tracking in long duration videos
* Live streaming of annotated videos to youtube live
* Visual AI from RTSP cameras


## Environment variables

The following environment varibles may be set by the client:

VIPY_RTSP_URL='rtsp://user@password:127.0.0.1'    
VIPY_YOUTUBE_STREAMKEY='xxxx-xxxx-xxxx-xxxx-xxxx'    
VIPY_CACHE='/home/username/.vipy'    

For additional environment variables, refer to the vipy package.


## Versioning

To determine what heyvi version you are running you can use:

>>> heyvi.__version__
>>> heyvi.version.is_at_least('0.0.6') 


# Contact

Visym Labs <info@visym.com>

"""

# Import all subpackages
import heyvi.version
import heyvi.system
import heyvi.recognition
import heyvi.detection
import heyvi.sensor

__version__ = heyvi.version.VERSION

