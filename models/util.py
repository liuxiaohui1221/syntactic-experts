import os
import time

from tqdm import tqdm

from ProjectPath import get_project_path
from models.ECSpell.Code.common_utils import load_json

test_data_path=os.path.join(get_project_path(),'models/macbert/output/final_val.json')

def eval_by_model(correct_fn, input_eval_path=test_data_path,returnType=1, verbose=True):
    """
    句级评估结果，设定需要纠错为正样本，无需纠错为负样本
    Args:
        correct_fn:
        input_eval_path:
        output_eval_path:
        verbose:

    Returns:
        Acc, Recall, F1
    """
    corpus = load_json(input_eval_path)
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    start_time = time.time()
    for data_dict in tqdm(corpus):
        src = data_dict.get('source', '')
        tgt = data_dict.get('target', '')
        errors = data_dict.get('errors', [])

        #  pred_detail: list(wrong, right, begin_idx, end_idx)
        if returnType==1:
            tgt_pred, pred_detail = correct_fn(src)
        else:
            res=correct_fn(src)
            tgt_pred, pred_detail = res[0],res[1]
        if verbose:
            print()
            print('input  :', src)
            print('truth  :', tgt, errors)
            print('predict:', tgt_pred, pred_detail)

        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == tgt_pred:
                TN += 1
                # print('right')
            # 预测为正
            else:
                FP += 1
                # print('wrong')
        # 正样本
        else:
            # 预测也为正
            if tgt == tgt_pred:
                TP += 1
                # print('right')
            # 预测为负
            else:
                FN += 1
                # print('wrong')
        total_num += 1
    spend_time = time.time() - start_time
    acc = (TP + TN) / (total_num+0.00001)
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    print(
        f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, '
        f'cost time:{spend_time:.2f} s, total num: {total_num}')
    return acc, precision, recall, f1
