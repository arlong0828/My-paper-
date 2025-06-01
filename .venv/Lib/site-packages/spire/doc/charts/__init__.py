import sys
from ctypes import *
from spire.doc.common import *

from spire.doc.common.Common import IntPtrArray
from spire.doc.common.Common import IntPtrWithTypeName

from spire.doc.common.SpireObject import SpireObject
from spire.doc.common.RectangleF import RectangleF

from spire.doc.charts.ApiException import ApiException
from spire.doc.charts.AxisBuiltInUnit import AxisBuiltInUnit
from spire.doc.charts.AxisCategoryType import AxisCategoryType
from spire.doc.charts.AxisCrosses import AxisCrosses
from spire.doc.charts.AxisScaleType import AxisScaleType
from spire.doc.charts.AxisTickLabelPosition import AxisTickLabelPosition
from spire.doc.charts.AxisTickMark import AxisTickMark
from spire.doc.charts.AxisTimeUnit import AxisTimeUnit
from spire.doc.charts.Chart import Chart
from spire.doc.charts.ChartAxis import ChartAxis
from spire.doc.charts.ChartDataLabel import ChartDataLabel
from spire.doc.charts.ChartDataLabelCollection import ChartDataLabelCollection
from spire.doc.charts.ChartDataPoint import ChartDataPoint
from spire.doc.charts.ChartDataPointCollection import ChartDataPointCollection
from spire.doc.charts.ChartLegend import ChartLegend
from spire.doc.charts.ChartMarker import ChartMarker
from spire.doc.charts.ChartNumberFormat import ChartNumberFormat
from spire.doc.charts.ChartSeries import ChartSeries
from spire.doc.charts.ChartSeriesCollection import ChartSeriesCollection
from spire.doc.charts.ChartTitle import ChartTitle
from spire.doc.charts.ChartType import ChartType
from spire.doc.charts.LegendPosition import LegendPosition
from spire.doc.charts.MarkerSymbol import MarkerSymbol