from tqdm import tqdm
import json
import numpy
from models.model_MiduCTC.src import corrector
import pycorrector
# testa_data = json.load(open('../data/preliminary_a_data/preliminary_a_test_source.json',encoding='utf-8'))
testa_data = json.load(open('../data/preliminary_a_data/preliminary_val.json',encoding='utf-8'))
correct= corrector.Corrector('../model/ctc_2022Y07M22D23H/epoch2,step1,testf1_39_94%,devf1_39_94%')


def replaceAndCheckAgain(source,tuple4):
    new_token=tuple4[1]
    pos_start=tuple4[2]
    pos_end = tuple4[3]
    new_target=source[:pos_start]+new_token+source[pos_end:]
    # print(source,new_target)
    corrected_sent=correct(new_target)[0]
    if corrected_sent==new_target:
        return True,new_target
    else:
        return False,new_target


def canIgnore(corrected_sent):
    pass

submit = []
idx=0
total_neg,neg_predict_pos,pyc_checked_neg=0,0,0
ctc_double_checked_err=0
for ins in tqdm(testa_data[:]):
    total_neg+=1
    # CTC模型预测
    corrected_sent = correct(ins['source'])[0]
    if corrected_sent==ins['source']:
        neg_predict_pos+=1

    corrected_sent2, detail = pycorrector.correct(ins['source'])

    # 1.CTC判正，pycorrector判错的样本，根据pycorrector纠错词替换(满足任意一个替换即可)后，
    # 再次通过CTC语义判断，若也为正，则将原样本标记为错例？按pycorrector的纠正方式处理。
    # 注意：（未实现）排除误检，即不考虑pycor检测的词语在原文分别前后组成词语(查找词语知识库)。
    if corrected_sent == ins['source'] and corrected_sent2 != ins['source']:
        # detail list(wrong, right, begin_idx, end_idx)
        choosed=False
        for tuple4 in detail:
            # 逐一替换和CTC再次验证合格
            isPostive,corrected2_postive = replaceAndCheckAgain(corrected_sent,tuple4)
            if isPostive:
            # 再次验证此时是否预测正确
                if corrected2_postive==ins['target']:
                    pyc_checked_neg+=1
                    submit.append({
                        "inference": corrected2_postive,
                        "id": ins['id']
                    })
                    choosed=True
                    break
        if choosed==False:
            ctc_double_checked_err += 1
            submit.append({
                "inference": corrected_sent,
                "id": ins['id']
            })
    # todo 2.CTC判错加“的”的可以忽略，在“已”后加“经”字的可以忽略
    elif corrected_sent==None:
        print("Ignore:",ins['source'])
        submit.append({
            "inference": ins['source'],
            "id": ins['id']
        })
    # TODO 3.首先对实体进行检测，对待替换词为实体的忽略。----优化pycorrector
    else:
        submit.append({
            "inference": corrected_sent,
            "id": ins['id']
        })
    idx += 1

print(total_neg,neg_predict_pos,ctc_double_checked_err,pyc_checked_neg)
# print(submit)
json.dump(submit, open('../data/preliminary_a_data/output/preliminary_a_test_source.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

