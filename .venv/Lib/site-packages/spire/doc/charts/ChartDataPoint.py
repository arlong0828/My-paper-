from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class ChartDataPoint (SpireObject) :
    """

    """
    @property
    def Index(self)->int:
        """

        """
        GetDllLibDoc().ChartDataPoint_get_Index.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataPoint_get_Index.restype=c_int
        ret = CallCFunction(GetDllLibDoc().ChartDataPoint_get_Index,self.Ptr)
        return ret

    @property
    def Explosion(self)->int:
        """

        """
        GetDllLibDoc().ChartDataPoint_get_Explosion.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataPoint_get_Explosion.restype=c_int
        ret = CallCFunction(GetDllLibDoc().ChartDataPoint_get_Explosion,self.Ptr)
        return ret

    @Explosion.setter
    def Explosion(self, value:int):
        GetDllLibDoc().ChartDataPoint_set_Explosion.argtypes=[c_void_p, c_int]
        CallCFunction(GetDllLibDoc().ChartDataPoint_set_Explosion,self.Ptr, value)

    @property
    def InvertIfNegative(self)->bool:
        """

        """
        GetDllLibDoc().ChartDataPoint_get_InvertIfNegative.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataPoint_get_InvertIfNegative.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartDataPoint_get_InvertIfNegative,self.Ptr)
        return ret

    @InvertIfNegative.setter
    def InvertIfNegative(self, value:bool):
        GetDllLibDoc().ChartDataPoint_set_InvertIfNegative.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartDataPoint_set_InvertIfNegative,self.Ptr, value)

    @property
    def Bubble3D(self)->bool:
        """

        """
        GetDllLibDoc().ChartDataPoint_get_Bubble3D.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataPoint_get_Bubble3D.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartDataPoint_get_Bubble3D,self.Ptr)
        return ret

    @Bubble3D.setter
    def Bubble3D(self, value:bool):
        GetDllLibDoc().ChartDataPoint_set_Bubble3D.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartDataPoint_set_Bubble3D,self.Ptr, value)

    @property

    def Marker(self)->'ChartMarker':
        """

        """
        GetDllLibDoc().ChartDataPoint_get_Marker.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataPoint_get_Marker.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().ChartDataPoint_get_Marker,self.Ptr)
        ret = None if intPtr==None else ChartMarker(intPtr)
        return ret


