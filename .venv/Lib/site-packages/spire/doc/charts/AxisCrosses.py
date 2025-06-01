from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class AxisCrosses(Enum):
    """
    <summary>
        Specifies the possible crossing points for an axis.
    </summary>
    """
    AutoZero = 0
    Max = 1
    Min = 2
    Custom = 3

