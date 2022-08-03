# 词性标注
import model.model_MiduCTC.src.thulac as thulac

class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls()
        return self._instance[self._cls]

# @Singleton
class ThulacSingle:
    def __init__(self):
        self.thu1 = thulac.thulac(seg_only=True)  # 默认模式
        pass