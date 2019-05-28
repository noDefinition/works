from .v1 import V1
from .n1 import N1
from .n5 import N5
from .n6 import N6
from .n7 import N7
from .n8 import N8
from .n10 import N10

name2m_class = {v.__name__: v for v in [N6]}
