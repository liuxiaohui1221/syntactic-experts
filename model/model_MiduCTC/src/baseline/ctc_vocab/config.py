import os

import jieba

from ProjectPath import get_project_path
from model.model_MiduCTC.src.baseline.ctc_vocab.thulac_singleton import ThulacSingle
class VocabConf:
    detect_vocab_size = 2
    correct_vocab_size = 20675

    vocab_types = ["unknow", "n", "np", "ns", "ni", "nz", "m", "q", "mq", "t", "f", "s", "v", "a", "d", "h", "k", "i",
                   "j",
                   "r",
                   "c", "p", "u", "y"
        , "e", "o", "g", "w", "x"]
    vocab_id2type = {"[unused" + str(i + 10) + "]": v for i, v in enumerate(vocab_types)}
    vocab_type2id = {v: "[unused" + str(i + 10) + "]" for i, v in enumerate(vocab_types)}
    def __init__(self):
        try:
            # self.thulac_singleton = ThulacSingle().thu1

            user_dict = "knowledgebase/dict/custom_dict.dic"
            user_dict = os.path.join(get_project_path(), user_dict)
            jieba.load_userdict(user_dict)
            self.jieba_singleton=jieba
        except Exception as e:
            print(e)