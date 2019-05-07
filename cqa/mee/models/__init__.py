from .v1 import V1
from .v3 import V3
from .v5 import V5
from .p1 import P1, P1m, R1
from .p2 import P2
from .p3 import P3
from .u1 import U1
from .w1 import W1

from .t1 import T1
from .s1 import S1
from .s2 import S2

from .b1 import B1
from ..base.common import CCC
from ..base.aaai15 import AAAI15
from ..base.aaai17 import AAAI17
from ..base.ijcai15 import IJCAI15

name2m_class = {v.__name__: v for v in [P1, S2, B1, AAAI15, AAAI17, IJCAI15]}
