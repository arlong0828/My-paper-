from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class Chart (SpireObject) :
    """
    <summary>
        Provides access to the chart shape properties.
    </summary>
    """
    @property

    def Series(self)->'ChartSeriesCollection':
        """
    <summary>
        Provides access to series collection.
    </summary>
        """
        GetDllLibDoc().Chart_get_Series.argtypes=[c_void_p]
        GetDllLibDoc().Chart_get_Series.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().Chart_get_Series,self.Ptr)
        from spire.doc.charts import ChartSeriesCollection
        ret = None if intPtr==None else ChartSeriesCollection(intPtr)
        return ret


    @property

    def Title(self)->'ChartTitle':
        """
    <summary>
         Provides access to the chart title properties.
    </summary>
        """
        GetDllLibDoc().Chart_get_Title.argtypes=[c_void_p]
        GetDllLibDoc().Chart_get_Title.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().Chart_get_Title,self.Ptr)
        from spire.doc.charts import ChartTitle
        ret = None if intPtr==None else ChartTitle(intPtr)
        return ret


    @property

    def Legend(self)->'ChartLegend':
        """
    <summary>
        Provides access to the chart legend properties.
    </summary>
        """
        GetDllLibDoc().Chart_get_Legend.argtypes=[c_void_p]
        GetDllLibDoc().Chart_get_Legend.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().Chart_get_Legend,self.Ptr)
        from spire.doc.charts import ChartLegend
        ret = None if intPtr==None else ChartLegend(intPtr)
        return ret


    @property

    def AxisX(self)->'ChartAxis':
        """
    <summary>
        Provides access to properties of the X axis of the chart.
    </summary>
        """
        GetDllLibDoc().Chart_get_AxisX.argtypes=[c_void_p]
        GetDllLibDoc().Chart_get_AxisX.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().Chart_get_AxisX,self.Ptr)
        from spire.doc.charts import ChartAxis
        ret = None if intPtr==None else ChartAxis(intPtr)
        return ret


    @property

    def AxisY(self)->'ChartAxis':
        """
    <summary>
        Provides access to properties of the Y axis of the chart.
    </summary>
        """
        GetDllLibDoc().Chart_get_AxisY.argtypes=[c_void_p]
        GetDllLibDoc().Chart_get_AxisY.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().Chart_get_AxisY,self.Ptr)
        from spire.doc.charts import ChartAxis
        ret = None if intPtr==None else ChartAxis(intPtr)
        return ret


    @property

    def AxisZ(self)->'ChartAxis':
        """
    <summary>
        Provides access to properties of the Z axis of the chart.
    </summary>
        """
        GetDllLibDoc().Chart_get_AxisZ.argtypes=[c_void_p]
        GetDllLibDoc().Chart_get_AxisZ.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().Chart_get_AxisZ,self.Ptr)
        from spire.doc.charts import ChartAxis
        ret = None if intPtr==None else ChartAxis(intPtr)
        return ret


