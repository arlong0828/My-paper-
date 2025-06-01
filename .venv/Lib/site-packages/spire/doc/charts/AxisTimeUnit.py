from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class AxisTimeUnit(Enum):
    """
    <summary>
        pecifies the unit of time for axes.
    </summary>
    """
    Auto = 0
    Days = 1
    Months = 2
    Years = 3

