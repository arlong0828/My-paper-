from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class ChartNumberFormat (SpireObject) :
    """

    """
    @property

    def FormatCode(self)->str:
        """

        """
        GetDllLibDoc().ChartNumberFormat_get_FormatCode.argtypes=[c_void_p]
        GetDllLibDoc().ChartNumberFormat_get_FormatCode.restype=c_char_p
        ret = CallCFunction(GetDllLibDoc().ChartNumberFormat_get_FormatCode,self.Ptr)
        return ret


    @FormatCode.setter
    def FormatCode(self, value:str):
        valuePtr = StrToPtr(value)
        GetDllLibDoc().ChartNumberFormat_set_FormatCode.argtypes=[c_void_p, c_char_p]
        CallCFunction(GetDllLibDoc().ChartNumberFormat_set_FormatCode,self.Ptr, valuePtr)

    @property
    def IsLinkedToSource(self)->bool:
        """

        """
        GetDllLibDoc().ChartNumberFormat_get_IsLinkedToSource.argtypes=[c_void_p]
        GetDllLibDoc().ChartNumberFormat_get_IsLinkedToSource.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartNumberFormat_get_IsLinkedToSource,self.Ptr)
        return ret

    @IsLinkedToSource.setter
    def IsLinkedToSource(self, value:bool):
        GetDllLibDoc().ChartNumberFormat_set_IsLinkedToSource.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartNumberFormat_set_IsLinkedToSource,self.Ptr, value)

