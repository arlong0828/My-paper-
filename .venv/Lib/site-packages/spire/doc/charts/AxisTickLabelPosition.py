from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class AxisTickLabelPosition(Enum):
    """
    <summary>
        Specifies the possible positions for tick labels.
    </summary>
    """
    High = 0
    Low = 1
    NextTo = 2
    none = 3
    Default = 2

