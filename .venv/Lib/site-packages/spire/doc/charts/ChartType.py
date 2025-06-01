from enum import Enum
from plum import dispatch
from typing import TypeVar,Union,Generic,List,Tuple
from spire.doc.common import *
from spire.doc.charts import *
from spire.doc import *
from ctypes import *
import abc

class ChartType(Enum):
    """
    <summary>
        Specifies type of a chart.
    </summary>
    """
    Area = 0
    AreaStacked = 1
    AreaPercentStacked = 2
    Area3D = 3
    Area3DStacked = 4
    Area3DPercentStacked = 5
    Bar = 6
    BarStacked = 7
    BarPercentStacked = 8
    Bar3D = 9
    Bar3DStacked = 10
    Bar3DPercentStacked = 11
    Bubble = 12
    Bubble3D = 13
    Column = 14
    ColumnStacked = 15
    ColumnPercentStacked = 16
    Column3D = 17
    Column3DStacked = 18
    Column3DPercentStacked = 19
    Column3DClustered = 20
    Doughnut = 21
    Line = 22
    LineStacked = 23
    LinePercentStacked = 24
    Line3D = 25
    Pie = 26
    Pie3D = 27
    PieOfBar = 28
    PieOfPie = 29
    Radar = 30
    Scatter = 31
    Stock = 32
    Surface = 33
    Surface3D = 34
