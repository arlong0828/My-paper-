from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc.charts import *
from spire.doc import *
from ctypes import *
import abc

class ChartSeriesCollection (  SpireObject) :
    """

    """

    def get_Item(self ,index:int)->'ChartSeries':
        """

        """
        
        GetDllLibDoc().ChartSeriesCollection_get_Item.argtypes=[c_void_p ,c_int]
        GetDllLibDoc().ChartSeriesCollection_get_Item.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().ChartSeriesCollection_get_Item,self.Ptr, index)
        ret = None if intPtr==None else ChartSeries(intPtr)
        return ret



    def GetEnumerator(self)->'IEnumerator':
        """

        """
        GetDllLibDoc().ChartSeriesCollection_GetEnumerator.argtypes=[c_void_p]
        GetDllLibDoc().ChartSeriesCollection_GetEnumerator.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().ChartSeriesCollection_GetEnumerator,self.Ptr)
        ret = None if intPtr==None else IEnumerator(intPtr)
        return ret



    def RemoveAt(self ,index:int):
        """

        """
        
        GetDllLibDoc().ChartSeriesCollection_RemoveAt.argtypes=[c_void_p ,c_int]
        CallCFunction(GetDllLibDoc().ChartSeriesCollection_RemoveAt,self.Ptr, index)

    def Clear(self):
        """

        """
        GetDllLibDoc().ChartSeriesCollection_Clear.argtypes=[c_void_p]
        CallCFunction(GetDllLibDoc().ChartSeriesCollection_Clear,self.Ptr)

    @dispatch

    def Add(self ,seriesName:str,categories:List[str],values:List[float])->ChartSeries:
        """
    <summary>
        Adds new ChartSeries to this collection.
            Use this method to add series to any type of Bar, Column, Line and Surface charts.
    </summary>
        """
        seriesNamePtr = StrToPtr(seriesName)
        #arraycategories:ArrayTypecategories = ""
        countcategories = len(categories)
        ArrayTypecategories = c_char_p * countcategories
        arraycategories = ArrayTypecategories()
        for i in range(0, countcategories):
            arraycategories[i] = StrToPtr(categories[i])

        #arrayvalues:ArrayTypevalues = ""
        countvalues = len(values)
        ArrayTypevalues = c_double * countvalues
        arrayvalues = ArrayTypevalues()
        for i in range(0, countvalues):
            arrayvalues[i] = values[i]


        GetDllLibDoc().ChartSeriesCollection_Add.argtypes=[c_void_p ,c_char_p,ArrayTypecategories,c_int,ArrayTypevalues,c_int]
        GetDllLibDoc().ChartSeriesCollection_Add.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().ChartSeriesCollection_Add,self.Ptr, seriesNamePtr,arraycategories,countcategories,arrayvalues,countvalues)
        ret = None if intPtr==None else ChartSeries(intPtr)
        return ret


    @dispatch

    def Add(self ,seriesName:str,xValues:List[float],yValues:List[float])->ChartSeries:
        """
    <summary>
        Adds new ChartSeries to this collection.
            Use this method to add series to any type of Scatter charts.
    </summary>
        """
        seriesNamePtr = StrToPtr(seriesName)
        #arrayxValues:ArrayTypexValues = ""
        countxValues = len(xValues)
        ArrayTypexValues = c_double * countxValues
        arrayxValues = ArrayTypexValues()
        for i in range(0, countxValues):
            arrayxValues[i] = xValues[i]

        #arrayyValues:ArrayTypeyValues = ""
        countyValues = len(yValues)
        ArrayTypeyValues = c_double * countyValues
        arrayyValues = ArrayTypeyValues()
        for i in range(0, countyValues):
            arrayyValues[i] = yValues[i]


        GetDllLibDoc().ChartSeriesCollection_AddSXY.argtypes=[c_void_p ,c_char_p,ArrayTypexValues,c_int,ArrayTypeyValues,c_int]
        GetDllLibDoc().ChartSeriesCollection_AddSXY.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().ChartSeriesCollection_AddSXY,self.Ptr, seriesNamePtr,arrayxValues,countxValues,arrayyValues,countyValues)
        ret = None if intPtr==None else ChartSeries(intPtr)
        return ret


#    @dispatch
#
#    def Add(self ,seriesName:str,dates:'DateTime[]',values:List[float])->ChartSeries:
#        """
#    <summary>
#        Adds new ChartSeries to this collection.
#            Use this method to add series to any type of Area, Radar and Stock charts.
#    </summary>
#        """
#        #arraydates:ArrayTypedates = ""
#        countdates = len(dates)
#        ArrayTypedates = c_void_p * countdates
#        arraydates = ArrayTypedates()
#        for i in range(0, countdates):
#            arraydates[i] = dates[i].Ptr
#
#        #arrayvalues:ArrayTypevalues = ""
#        countvalues = len(values)
#        ArrayTypevalues = c_double * countvalues
#        arrayvalues = ArrayTypevalues()
#        for i in range(0, countvalues):
#            arrayvalues[i] = values[i]
#
#
#        GetDllLibDoc().ChartSeriesCollection_AddSDV.argtypes=[c_void_p ,c_wchar_p,ArrayTypedates,ArrayTypevalues]
#        GetDllLibDoc().ChartSeriesCollection_AddSDV.restype=c_void_p
#        intPtr = CallCFunction(GetDllLibDoc().ChartSeriesCollection_AddSDV,self.Ptr, seriesName,arraydates,arrayvalues)
#        ret = None if intPtr==None else ChartSeries(intPtr)
#        return ret
#


    @dispatch

    def Add(self ,seriesName:str,xValues:List[float],yValues:List[float],bubbleSizes:List[float])->ChartSeries:
        """
    <summary>
        Adds new ChartSeries to this collection.
            Use this method to add series to any type of Bubble charts.
    </summary>
        """
        seriesNamePtr = StrToPtr(seriesName)
        #arrayxValues:ArrayTypexValues = ""
        countxValues = len(xValues)
        ArrayTypexValues = c_double * countxValues
        arrayxValues = ArrayTypexValues()
        for i in range(0, countxValues):
            arrayxValues[i] = xValues[i]

        #arrayyValues:ArrayTypeyValues = ""
        countyValues = len(yValues)
        ArrayTypeyValues = c_double * countyValues
        arrayyValues = ArrayTypeyValues()
        for i in range(0, countyValues):
            arrayyValues[i] = yValues[i]

        #arraybubbleSizes:ArrayTypebubbleSizes = ""
        countbubbleSizes = len(bubbleSizes)
        ArrayTypebubbleSizes = c_double * countbubbleSizes
        arraybubbleSizes = ArrayTypebubbleSizes()
        for i in range(0, countbubbleSizes):
            arraybubbleSizes[i] = bubbleSizes[i]


        GetDllLibDoc().ChartSeriesCollection_AddSXYB.argtypes=[c_void_p ,c_char_p,ArrayTypexValues,c_int,ArrayTypeyValues,c_int,ArrayTypebubbleSizes,c_int]
        GetDllLibDoc().ChartSeriesCollection_AddSXYB.restype=c_void_p
        intPtr = CallCFunction(GetDllLibDoc().ChartSeriesCollection_AddSXYB,self.Ptr, seriesNamePtr,arrayxValues,countxValues,arrayyValues,countyValues,arraybubbleSizes,countbubbleSizes)
        ret = None if intPtr==None else ChartSeries(intPtr)
        return ret


    @property
    def Count(self)->int:
        """

        """
        GetDllLibDoc().ChartSeriesCollection_get_Count.argtypes=[c_void_p]
        GetDllLibDoc().ChartSeriesCollection_get_Count.restype=c_int
        ret = CallCFunction(GetDllLibDoc().ChartSeriesCollection_get_Count,self.Ptr)
        return ret

