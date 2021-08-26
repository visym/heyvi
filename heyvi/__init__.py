"""
Hey Vi 

## Versioning

To determine what heyvi version you are running you can use:

>>> heyvi.__version__
>>> heyvi.version.is_at_least('1.11.1') 

# Contact

Visym Labs <info@visym.com>

"""

# Import all subpackages
import heyvi.version

__version__ = heyvi.version.VERSION

