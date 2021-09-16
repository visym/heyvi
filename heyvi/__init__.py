"""

# \"Hey Vi!\"

## Environment variables

VIPY_RTSP_URL='rtsp://user@password:127.0.0.1'  
VIPY_YOUTUBE_STREAMKEY='xxxx-xxxx-xxxx-xxxx-xxxx'
VIPY_CACHE='/home/username/.vipy'

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

