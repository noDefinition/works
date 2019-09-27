from uclu.me.models.v1 import V1
from uclu.me.models.v2 import V2
from uclu.me.models.v3 import V3
from uclu.me.models.uatt import UAtt

m_classes = [V1, V2, V3, UAtt]
name2m_class = {m.__name__: m for m in m_classes}
