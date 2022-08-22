from models.mypycorrector.t5.t5_corrector import T5Corrector
from models.util import eval_by_model


if __name__ == "__main__":
    m = T5Corrector(model_dir='pretrained/checkpoint-5000')
    print(m.t5_correct('长沙地网络'))
    # eval_by_model(m.t5_correct,verbose=True)