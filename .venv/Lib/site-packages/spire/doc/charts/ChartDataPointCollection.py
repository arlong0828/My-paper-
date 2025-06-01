from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class ChartDataPointCollection (  IEnumerable) :
    """

    """

    def get_Item(self ,index:int)->'ChartDataPoint':
        """

        """
        
        GetDllLibDoc().ChartDataPointCollection_get_Item.argtypes=[c_void_p ,c_int]
        GetDllLibDoc().ChartDataPointCollection_get_Item.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().ChartDataPointCollection_get_Item,self.Ptr, index)
        ret = None if intPtr==None else ChartDataPoint(intPtr)
        return ret



    def GetEnumerator(self)->'IEnumerator':
        """

        """
        GetDllLibDoc().ChartDataPointCollection_GetEnumerator.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataPointCollection_GetEnumerator.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().ChartDataPointCollection_GetEnumerator,self.Ptr)
        ret = None if intPtr==None else IEnumerator(intPtr)
        return ret



    def Add(self ,index:int)->'ChartDataPoint':
        """

        """
        
        GetDllLibDoc().ChartDataPointCollection_Add.argtypes=[c_void_p ,c_int]
        GetDllLibDoc().ChartDataPointCollection_Add.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().ChartDataPointCollection_Add,self.Ptr, index)
        ret = None if intPtr==None else ChartDataPoint(intPtr)
        return ret



    def RemoveAt(self ,index:int):
        """

        """
        
        GetDllLibDoc().ChartDataPointCollection_RemoveAt.argtypes=[c_void_p ,c_int]
        CallCFunction(GetDllLibDoc().ChartDataPointCollection_RemoveAt,self.Ptr, index)

    def Clear(self):
        """

        """
        GetDllLibDoc().ChartDataPointCollection_Clear.argtypes=[c_void_p]
        CallCFunction(GetDllLibDoc().ChartDataPointCollection_Clear,self.Ptr)

    @property
    def Count(self)->int:
        """

        """
        GetDllLibDoc().ChartDataPointCollection_get_Count.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataPointCollection_get_Count.restype=c_int
        ret = CallCFunction(GetDllLibDoc().ChartDataPointCollection_get_Count,self.Ptr)
        return ret

