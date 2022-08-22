import torch
from tqdm import tqdm
import json

from models.macbert.util.common import getEdits, getTextEdits
from models.model_MiduCTC.src import corrector, correctorV3
from tqdm import tqdm
import json
import numpy
from models.model_MiduCTC.src import corrector
# import mypycorrector
# testa_data = json.load(open('../data/preliminary_a_data/preliminary_a_test_source.json',encoding='utf-8'))
testa_data = json.load(open('../data/preliminary_a_data/preliminary_val.json',encoding='utf-8'))
# 模型
# correct= corrector.Corrector('../new_model/ctc_2022Y07M21D01H/epoch2,step62500,testf1_35_77%,devf1_35_77%')
# correct= corrector.Corrector('../new_model/ctc_2022Y07M21D08H/epoch4,step1,testf1_35_94%,devf1_35_94%')
correct= corrector.Corrector('../pretrained_model/epoch3,step1,testf1_61_91%,devf1_55_17%')
submit = []
idx=0
equ_nums=0
s1,s2=0,0
exceed_max=0
print([1]+[2]*2)


def getLossWord(src, tar):
    edits=getTextEdits(src,tar)
    losswords=[]
    for edit in edits:
        if edit[0]!='insert':
            continue
        losswords.append((edit[1], src[edit[1]], tar[edit[3]:edit[4]]))
    return losswords

loss_count=0
pred_succ=0
recall_succ=0
recall_err=0
uncorrect=0
def existsLossWordInRecallSets(loss_words, pred_outputs):
    # 依次查找loss word
    for lossword in loss_words:
        word_recall_set=pred_outputs[0][lossword[0]][1]
        exist_flag=False
        for edit_word in word_recall_set:
            if "$APPEND" not in edit_word:
                continue
            candidate_insert_word=edit_word.split('_')[-1]
            actual_insert_word=lossword[2]
            if candidate_insert_word==actual_insert_word:
                exist_flag=True
                break
        if exist_flag==False:
            return False
    return True


for ins in tqdm(testa_data[:]):
    # 获得缺字样本
    loss_words=getLossWord(ins['source'],ins['target'])
    if len(loss_words)==0:
        continue
    loss_count+=1
    corrected_sent,pred_outputs = correct.recall(ins['source'],return_topk=5)
    if corrected_sent[0]==ins['target']:
        pred_succ+=1
    else:
        if corrected_sent[0]==ins['source']:
            uncorrect+=1
        # 判断所缺字是否在召回集中
        flag = existsLossWordInRecallSets(loss_words,pred_outputs)
        if flag:
            recall_succ+=1
        else:
            recall_err+=1
    print(corrected_sent)
    submit.append({
        "inference": corrected_sent[0],
        "id": ins['id']
    })
    idx += 1
print("loss_count, uncorrect, pred_succ, recall_succ, recall_err:",loss_count,uncorrect,pred_succ,recall_succ,recall_err)
print("exceed,total nums:",exceed_max,idx)
json.dump(submit, open('../data/preliminary_a_data/output/preliminary_a_test_source.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

