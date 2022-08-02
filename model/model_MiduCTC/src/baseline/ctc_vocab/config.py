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
            self.thulac_singleton = ThulacSingle().thu1
        except Exception as e:
            print(e)