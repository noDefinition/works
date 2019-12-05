from uclu.bert.models.b1 import B1
from uclu.bert.models.suv1 import SUV1
# names = set()
# for name in dir():
#     if name[0].isupper():
#         names.add(name)
# print(names)

m_classes = [eval(name) for name in dir() if name[0].isupper()]
name2m_class = {m.__name__: m for m in m_classes}
print(name2m_class)
