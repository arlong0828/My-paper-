from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class ChartTitle (SpireObject) :
    """

    """
    @property

    def Text(self)->str:
        """

        """
        GetDllLibDoc().ChartTitle_get_Text.argtypes=[c_void_p]
        GetDllLibDoc().ChartTitle_get_Text.restype=c_char_p
        ret = CallCFunction(GetDllLibDoc().ChartTitle_get_Text,self.Ptr)
        return ret


    @Text.setter
    def Text(self, value:str):
        textPtr = StrToPtr(value)
        GetDllLibDoc().ChartTitle_set_Text.argtypes=[c_void_p, c_char_p]
        CallCFunction(GetDllLibDoc().ChartTitle_set_Text,self.Ptr, textPtr)

    @property
    def Overlay(self)->bool:
        """

        """
        GetDllLibDoc().ChartTitle_get_Overlay.argtypes=[c_void_p]
        GetDllLibDoc().ChartTitle_get_Overlay.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartTitle_get_Overlay,self.Ptr)
        return ret

    @Overlay.setter
    def Overlay(self, value:bool):
        GetDllLibDoc().ChartTitle_set_Overlay.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartTitle_set_Overlay,self.Ptr, value)

    @property
    def Show(self)->bool:
        """

        """
        GetDllLibDoc().ChartTitle_get_Show.argtypes=[c_void_p]
        GetDllLibDoc().ChartTitle_get_Show.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartTitle_get_Show,self.Ptr)
        return ret

    @Show.setter
    def Show(self, value:bool):
        GetDllLibDoc().ChartTitle_set_Show.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartTitle_set_Show,self.Ptr, value)

