
from typing import List

from models.model_MiduCTC.src.baseline.predictor import PredictorCtc


class Corrector:
    def __init__(self, in_model_dir:str,ctc_label_vocab_dir='./baseline/ctc_vocab'):
        """_summary_

        Args:
            in_model_dir (str): 训练好的模型目录
        """
        self._predictor = PredictorCtc(
        in_model_dir=in_model_dir,
        ctc_label_vocab_dir=ctc_label_vocab_dir,
        use_cuda=True,
        cuda_id=None,
    )
        
    
    def __call__(self, texts:List[str]) -> List[str]:
        pred_outputs = self._predictor.predict(texts,return_topk=3)
        # print(pred_outputs)
        pred_texts = [PredictorCtc.output2text(output) for output in pred_outputs]
        return pred_texts
    def getCorrectedByPredOutputs(self,pred_outputs):
        return [PredictorCtc.output2text(output) for output in pred_outputs]

    def recall(self, texts:List[str],return_topk=20):
        pred_outputs = self._predictor.predict(texts, return_topk=return_topk)
        # print(pred_outputs)
        pred_texts = [PredictorCtc.output2text(output) for output in pred_outputs]
        return pred_texts,pred_outputs
    