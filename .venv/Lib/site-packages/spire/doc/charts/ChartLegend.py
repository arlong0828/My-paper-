from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class ChartLegend (SpireObject) :
    """

    """
    @property

    def Position(self)->'LegendPosition':
        """

        """
        GetDllLibDoc().ChartLegend_get_Position.argtypes=[c_void_p]
        GetDllLibDoc().ChartLegend_get_Position.restype=c_int
        ret = CallCFunction(GetDllLibDoc().ChartLegend_get_Position,self.Ptr)
        objwraped = LegendPosition(ret)
        return objwraped

    @Position.setter
    def Position(self, value:'LegendPosition'):
        GetDllLibDoc().ChartLegend_set_Position.argtypes=[c_void_p, c_int]
        CallCFunction(GetDllLibDoc().ChartLegend_set_Position,self.Ptr, value.value)

    @property
    def Overlay(self)->bool:
        """

        """
        GetDllLibDoc().ChartLegend_get_Overlay.argtypes=[c_void_p]
        GetDllLibDoc().ChartLegend_get_Overlay.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartLegend_get_Overlay,self.Ptr)
        return ret

    @Overlay.setter
    def Overlay(self, value:bool):
        GetDllLibDoc().ChartLegend_set_Overlay.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartLegend_set_Overlay,self.Ptr, value)

