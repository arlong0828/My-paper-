from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class AxisTickMark(Enum):
    """
    <summary>
        Specifies the possible positions for tick marks.
    </summary>
    """
    Cross = 0
    Inside = 1
    Outside = 2
    none = 3

