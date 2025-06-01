from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class MarkerSymbol(Enum):
    """
    <summary>
        Specifies marker symbol style.
    </summary>
    """
    Default = 0
    Circle = 1
    Dash = 2
    Diamond = 3
    Dot = 4
    none = 5
    Picture = 6
    Plus = 7
    Square = 8
    Star = 9
    Triangle = 10
    X = 11

