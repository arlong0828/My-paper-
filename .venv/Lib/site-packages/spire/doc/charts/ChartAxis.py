from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class ChartAxis (SpireObject) :
    """

    """
    @property

    def NumberFormat(self)->'ChartNumberFormat':
        """

        """
        GetDllLibDoc().ChartAxis_get_NumberFormat.argtypes=[c_void_p]
        GetDllLibDoc().ChartAxis_get_NumberFormat.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().ChartAxis_get_NumberFormat,self.Ptr)
        from spire.doc.charts import ChartNumberFormat
        ret = None if intPtr==None else ChartNumberFormat(intPtr)
        return ret


