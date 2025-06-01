from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class LegendPosition(Enum):
    """
    <summary>
        Specifies the possible positions for a chart legend.
    </summary>
    """
    none = 0
    Bottom = 1
    Left = 2
    Right = 3
    Top = 4
    TopRight = 5

