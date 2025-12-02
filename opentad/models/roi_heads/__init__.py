from .standard_roi_head import StandardRoIHead
from .cascade_roi_head import CascadeRoIHead
from .standard_map_head import StandardProposalMapHead
from .etad_roi_head import ETADRoIHead
# 延迟导入AFSD和VSGN，因为它们依赖需要编译的扩展模块
try:
    from .afsd_roi_head import AFSDRefineHead
except ImportError:
    AFSDRefineHead = None
try:
    from .vsgn_roi_head import VSGNRoIHead
except ImportError:
    VSGNRoIHead = None

from .proposal_generator import *
# 延迟导入roi_extractors，因为它们可能依赖需要编译的扩展模块
try:
    from .roi_extractors import *
except ImportError:
    pass
from .proposal_head import *

__all__ = [
    "StandardRoIHead",
    "CascadeRoIHead",
    "StandardProposalMapHead",
    "ETADRoIHead",
    "AFSDRefineHead",
    "VSGNRoIHead",
]
