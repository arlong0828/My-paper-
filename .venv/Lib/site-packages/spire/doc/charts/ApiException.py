from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class ApiException (SpireObject) :
    """

    """
    @property
    def ErrorCode(self)->int:
        """

        """
        GetDllLibDoc().ApiException_get_ErrorCode.argtypes=[c_void_p]
        GetDllLibDoc().ApiException_get_ErrorCode.restype=c_int
        ret = CallCFunction(GetDllLibDoc().ApiException_get_ErrorCode,self.Ptr)
        return ret

    @ErrorCode.setter
    def ErrorCode(self, value:int):
        GetDllLibDoc().ApiException_set_ErrorCode.argtypes=[c_void_p, c_int]
        CallCFunction(GetDllLibDoc().ApiException_set_ErrorCode,self.Ptr, value)

    @property

    def ErrorContent(self)->'SpireObject':
        """

        """
        GetDllLibDoc().ApiException_get_ErrorContent.argtypes=[c_void_p]
        GetDllLibDoc().ApiException_get_ErrorContent.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().ApiException_get_ErrorContent,self.Ptr)
        ret = None if intPtr==None else SpireObject(intPtr)
        return ret


