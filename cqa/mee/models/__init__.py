# from .v1 import V1
# from .v3 import V3
# from .v5 import V5

from cqa.mee.models.p1 import P1
from cqa.mee.models.p2 import P2
from cqa.mee.models.p3 import P3
from cqa.mee.models.p4 import P4

# from .u1 import U1
# from .w1 import W1
# from .t1 import T1
# from .s1 import S1
# from .s2 import S2
from cqa.mee.models.b1 import B1

from cqa.mee.base.common import CqaBaseline
from cqa.mee.base.aaai15 import AAAI15
from cqa.mee.base.aaai17 import AAAI17
from cqa.mee.base.ijcai15 import IJCAI15

name2m_class = {v.__name__: v for v in [P1, P2, P3, P4, AAAI15, AAAI17, IJCAI15]}
