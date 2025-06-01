from enum import Enum
from plum import dispatch
from typing import TypeVar, Union, Generic, List, Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class FileFormat(Enum):
    """
    Enum class representing different types of file formats.
    """
    Doc = 0
    Dot = 5
    Docx = 10
    Docx2010 = 11
    Docx2013 = 12
    Docx2016 = 13
    Docx2019 = 14
    Dotx = 20
    Dotx2010 = 21
    Dotx2013 = 22
    Dotx2016 = 23
    Dotx2019 = 24
    Docm = 30
    Docm2010 = 31
    Docm2013 = 32
    Docm2016 = 33
    Docm2019 = 34
    Dotm = 40
    Dotm2010 = 41
    Dotm2013 = 42
    Dotm2016 = 43
    Dotm2019 = 44
    OOXML = 50
    WordML = 60
    WordXml = 70
    Odt = 80
    Ott = 90
    PDF = 100
    Txt = 110
    Rtf = 120
    SVG = 130
    Xml = 140
    Mhtml = 150
    Html = 160
    XPS = 170
    EPub = 180
    DocPre97 = 190
    PostScript = 200
    PCL = 210
    OFD = 220
    OnlineDoc = 230
    Wps = 240
    Wpt = 250
    Markdown = 260
    Auto = 300
    