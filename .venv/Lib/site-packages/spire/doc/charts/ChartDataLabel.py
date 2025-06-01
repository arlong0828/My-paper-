from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class ChartDataLabel (SpireObject) :
    """

    """
    @property
    def Index(self)->int:
        """

        """
        GetDllLibDoc().ChartDataLabel_get_Index.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataLabel_get_Index.restype=c_int
        ret = CallCFunction(GetDllLibDoc().ChartDataLabel_get_Index,self.Ptr)
        return ret

    @property
    def ShowCategoryName(self)->bool:
        """

        """
        GetDllLibDoc().ChartDataLabel_get_ShowCategoryName.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataLabel_get_ShowCategoryName.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartDataLabel_get_ShowCategoryName,self.Ptr)
        return ret

    @ShowCategoryName.setter
    def ShowCategoryName(self, value:bool):
        GetDllLibDoc().ChartDataLabel_set_ShowCategoryName.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartDataLabel_set_ShowCategoryName,self.Ptr, value)

    @property
    def ShowBubbleSize(self)->bool:
        """

        """
        GetDllLibDoc().ChartDataLabel_get_ShowBubbleSize.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataLabel_get_ShowBubbleSize.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartDataLabel_get_ShowBubbleSize,self.Ptr)
        return ret

    @ShowBubbleSize.setter
    def ShowBubbleSize(self, value:bool):
        GetDllLibDoc().ChartDataLabel_set_ShowBubbleSize.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartDataLabel_set_ShowBubbleSize,self.Ptr, value)

    @property
    def ShowLegendKey(self)->bool:
        """

        """
        GetDllLibDoc().ChartDataLabel_get_ShowLegendKey.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataLabel_get_ShowLegendKey.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartDataLabel_get_ShowLegendKey,self.Ptr)
        return ret

    @ShowLegendKey.setter
    def ShowLegendKey(self, value:bool):
        GetDllLibDoc().ChartDataLabel_set_ShowLegendKey.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartDataLabel_set_ShowLegendKey,self.Ptr, value)

    @property
    def ShowPercentage(self)->bool:
        """

        """
        GetDllLibDoc().ChartDataLabel_get_ShowPercentage.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataLabel_get_ShowPercentage.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartDataLabel_get_ShowPercentage,self.Ptr)
        return ret

    @ShowPercentage.setter
    def ShowPercentage(self, value:bool):
        GetDllLibDoc().ChartDataLabel_set_ShowPercentage.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartDataLabel_set_ShowPercentage,self.Ptr, value)

    @property
    def ShowSeriesName(self)->bool:
        """

        """
        GetDllLibDoc().ChartDataLabel_get_ShowSeriesName.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataLabel_get_ShowSeriesName.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartDataLabel_get_ShowSeriesName,self.Ptr)
        return ret

    @ShowSeriesName.setter
    def ShowSeriesName(self, value:bool):
        GetDllLibDoc().ChartDataLabel_set_ShowSeriesName.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartDataLabel_set_ShowSeriesName,self.Ptr, value)

    @property
    def ShowValue(self)->bool:
        """

        """
        GetDllLibDoc().ChartDataLabel_get_ShowValue.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataLabel_get_ShowValue.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartDataLabel_get_ShowValue,self.Ptr)
        return ret

    @ShowValue.setter
    def ShowValue(self, value:bool):
        GetDllLibDoc().ChartDataLabel_set_ShowValue.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartDataLabel_set_ShowValue,self.Ptr, value)

    @property
    def ShowLeaderLines(self)->bool:
        """

        """
        GetDllLibDoc().ChartDataLabel_get_ShowLeaderLines.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataLabel_get_ShowLeaderLines.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartDataLabel_get_ShowLeaderLines,self.Ptr)
        return ret

    @ShowLeaderLines.setter
    def ShowLeaderLines(self, value:bool):
        GetDllLibDoc().ChartDataLabel_set_ShowLeaderLines.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartDataLabel_set_ShowLeaderLines,self.Ptr, value)

    @property
    def ShowDataLabelsRange(self)->bool:
        """

        """
        GetDllLibDoc().ChartDataLabel_get_ShowDataLabelsRange.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataLabel_get_ShowDataLabelsRange.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartDataLabel_get_ShowDataLabelsRange,self.Ptr)
        return ret

    @ShowDataLabelsRange.setter
    def ShowDataLabelsRange(self, value:bool):
        GetDllLibDoc().ChartDataLabel_set_ShowDataLabelsRange.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().ChartDataLabel_set_ShowDataLabelsRange,self.Ptr, value)

    @property

    def Separator(self)->str:
        """

        """
        GetDllLibDoc().ChartDataLabel_get_Separator.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataLabel_get_Separator.restype=c_char_p
        ret = CallCFunction(GetDllLibDoc().ChartDataLabel_get_Separator,self.Ptr)
        return ret


    @Separator.setter
    def Separator(self, value:str):
        valuePtr = StrToPtr(value)
        GetDllLibDoc().ChartDataLabel_set_Separator.argtypes=[c_void_p, c_char_p]
        CallCFunction(GetDllLibDoc().ChartDataLabel_set_Separator,self.Ptr, valuePtr)

    @property
    def IsVisible(self)->bool:
        """

        """
        GetDllLibDoc().ChartDataLabel_get_IsVisible.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataLabel_get_IsVisible.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().ChartDataLabel_get_IsVisible,self.Ptr)
        return ret

    @property

    def NumberFormat(self)->'ChartNumberFormat':
        """

        """
        GetDllLibDoc().ChartDataLabel_get_NumberFormat.argtypes=[c_void_p]
        GetDllLibDoc().ChartDataLabel_get_NumberFormat.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().ChartDataLabel_get_NumberFormat,self.Ptr)
        ret = None if intPtr==None else ChartNumberFormat(intPtr)
        return ret


