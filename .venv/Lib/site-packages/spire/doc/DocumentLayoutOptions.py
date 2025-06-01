from enum import Enum
from plum import dispatch
from typing import TypeVar, Union, Generic, List, Tuple
from spire.doc.common import *
from spire.doc import *
from ctypes import *
import abc

class DocumentLayoutOptions (SpireObject) :
    """
    <summary>
        Represents the settings for managing the document layout procedure.
            The related layout settings are only supported for new engines.
    </summary>
    """
    @property
    def ShowHiddenText(self)->bool:
        """
    <summary>
        Gets or sets a value indicating whether is show hidden text.
            Default is false.
    </summary>
        """
        GetDllLibDoc().DocumentLayoutOptions_get_ShowHiddenText.argtypes=[c_void_p]
        GetDllLibDoc().DocumentLayoutOptions_get_ShowHiddenText.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().DocumentLayoutOptions_get_ShowHiddenText,self.Ptr)
        return ret

    @ShowHiddenText.setter
    def ShowHiddenText(self, value:bool):
        GetDllLibDoc().DocumentLayoutOptions_set_ShowHiddenText.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().DocumentLayoutOptions_set_ShowHiddenText,self.Ptr, value)

    @property
    def UseHarfBuzzTextShaper(self)->bool:
        """
    <summary>
        Gets or sets a value indicating whether use HarfBuzz text shaper.
            The default value is false.
    </summary>
        """
        GetDllLibDoc().DocumentLayoutOptions_get_UseHarfBuzzTextShaper.argtypes=[c_void_p]
        GetDllLibDoc().DocumentLayoutOptions_get_UseHarfBuzzTextShaper.restype=c_bool
        ret = CallCFunction(GetDllLibDoc().DocumentLayoutOptions_get_UseHarfBuzzTextShaper,self.Ptr)
        return ret

    @UseHarfBuzzTextShaper.setter
    def UseHarfBuzzTextShaper(self, value:bool):
        GetDllLibDoc().DocumentLayoutOptions_set_UseHarfBuzzTextShaper.argtypes=[c_void_p, c_bool]
        CallCFunction(GetDllLibDoc().DocumentLayoutOptions_set_UseHarfBuzzTextShaper,self.Ptr, value)

    @property

    def CommentDisplayMode(self)->'CommentDisplayMode':
        """
    <summary>
         Gets or sets the way comments are rendered.
             Default value is <see cref="F:Spire.Doc.Layout.CommentDisplayMode.ShowInBalloons" />.
    </summary>
        """
        GetDllLibDoc().DocumentLayoutOptions_get_CommentDisplayMode.argtypes=[c_void_p]
        GetDllLibDoc().DocumentLayoutOptions_get_CommentDisplayMode.restype=c_int
        ret = CallCFunction(GetDllLibDoc().DocumentLayoutOptions_get_CommentDisplayMode,self.Ptr)
        objwraped = CommentDisplayMode(ret)
        return objwraped

    @CommentDisplayMode.setter
    def CommentDisplayMode(self, value:'CommentDisplayMode'):
        GetDllLibDoc().DocumentLayoutOptions_set_CommentDisplayMode.argtypes=[c_void_p, c_int]
        CallCFunction(GetDllLibDoc().DocumentLayoutOptions_set_CommentDisplayMode,self.Ptr, value.value)

