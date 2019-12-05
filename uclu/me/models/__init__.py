from uclu.me.models.d2v import Doc2vec
from uclu.me.models.n1 import N1
from uclu.me.models.t1 import T1
from uclu.me.models.t2 import T2
from uclu.me.models.t3 import T3
from uclu.me.models.uatt import UAtt
from uclu.me.models.v1 import V1
from uclu.me.models.v2 import V2
from uclu.me.models.v3 import V3
from uclu.me.models.v4 import V4
from uclu.me.models.v5 import V5

m_classes = [eval(name) for name in dir() if name[0].isupper()]
name2m_class = {m.__name__: m for m in m_classes}
# print(name2m_class)
