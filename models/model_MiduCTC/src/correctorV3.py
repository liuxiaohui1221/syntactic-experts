
from typing import List

from models.model_MiduCTC.src.baseline.predictor import PredictorCtc
from models.model_MiduCTC.src.baseline.predictorV3 import PredictorCtcV3


class CorrectorV3:
    def __init__(self, in_model_dir:str):
        """_summary_

        Args:
            in_model_dir (str): 训练好的模型目录
        """
        self._predictor = PredictorCtcV3(
        in_model_dir=in_model_dir,
        ctc_label_vocab_dir='./baseline/ctc_vocab',
        use_cuda=True,
        cuda_id=None,
    )
        
    
    def __call__(self, texts:List[str]) -> List[str]:
        pred_outputs = self._predictor.predict(texts)
        # print(pred_outputs)
        pred_texts = [PredictorCtc.output2text(output) for output in pred_outputs]
        return pred_texts

    