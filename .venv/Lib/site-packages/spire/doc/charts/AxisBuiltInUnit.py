from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class AxisBuiltInUnit(Enum):
    """
    <summary>
         Specifies the display units for an axis.
    </summary>
    """
    none = 0
    Custom = 1
    Billions = 2
    HundredMillions = 3
    Hundreds = 4
    HundredThousands = 5
    Millions = 6
    TenMillions = 7
    TenThousands = 8
    Thousands = 9
    Trillions = 10
    Percentage = 11

