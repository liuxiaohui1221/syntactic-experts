import os.path

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
from models.mypycorrector.ModelPath import get_mypycorrector_path
from models.mypycorrector.t5.t5_corrector import T5Corrector

testa_data = json.load(open('../data/preliminary_a_data/preliminary_val.json',encoding='utf-8'))
# 模型
# correct= corrector.Corrector('../new_model/ctc_2022Y07M21D01H/epoch2,step62500,testf1_35_77%,devf1_35_77%')
# correct= corrector.Corrector('../new_model/ctc_2022Y07M21D08H/epoch4,step1,testf1_35_94%,devf1_35_94%')
correct= corrector.Corrector('../pretrained_model/epoch3,step1,testf1_61_91%,devf1_55_17%')
# loss_correct=corrector.Corrector('../model/epoch10,step1,testf1_12_12%,devf1_58_23%')

t5_path=os.path.join(get_mypycorrector_path(),'pretrained/checkpoint-5000')
m = T5Corrector(model_dir=t5_path)
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


def getCommonPosAndAppendingWords(pred_outputs, losspred_outputs):
    pos_apendingwords=[]
    stopwords=['']
    for index,word_candidates_tuple in enumerate(pred_outputs[0]):
        word_edits=word_candidates_tuple[1]
        losword_edits=losspred_outputs[0][index][1]
        # 过滤相同预测
        if word_edits[0]==losword_edits[0]:
            continue
        for i,word_edit in enumerate(word_edits):
            if "$APPEND" in word_edit:
                if word_edit in losword_edits:
                    # candidate_insert_word = word_edit.split('_')[-1]
                    # if candidate_insert_word in stopwords:
                    #     continue
                    pos_apendingwords.append((index,word_edit))
                    break
    if len(pos_apendingwords)>2 or len(pos_apendingwords)==0:
        return None
    print(pos_apendingwords)
    # 替换pred_outputs对应位置
    for pos_words in pos_apendingwords:
        pos_edits=pred_outputs[0][pos_words[0]][1]
        pos_edits[0]=pos_words[1]
    return pred_outputs

def getNewCorrected(comm_pos_and_apendingwords, pred_outputs):
    pass

before_succ,before_err=0,0


def existsCommonPosWords(pred_outputs, pred_detail):
    for edit in pred_detail:
        word_candidates_tuple=pred_outputs[0][edit[2]]
        判断是否存在相同修改字
    for index, word_candidates_tuple in enumerate(pred_outputs[0]):
        word_edits = word_candidates_tuple[1]


for ins in testa_data[:]:
    loss_count+=1
    corrected_sent,pred_outputs = correct.recall(ins['source'],return_topk=20)

    corrected_new=corrected_sent[0]
    if corrected_sent[0]!=ins['source']:
        uncorrect+=1
    else:
        # 缺字召回与正常召回恰存在1-2个交集的则当做缺字纠错
        tgt_pred, pred_detail=m.t5_correct(ins['source'])
        # losscorrected_sent, losspred_outputs = loss_correct.recall(ins['source'], return_topk=1)
        # 获得共同缺字样本集
        # new_pred_outputs=getCommonPosAndAppendingWords(pred_outputs,losspred_outputs)
        flag=existsCommonPosWords(pred_outputs,pred_detail)
        if new_pred_outputs==None:
            continue
        corrected_new=correct.getCorrectedByPredOutputs(new_pred_outputs)

        if corrected_new[0]==ins['target']:
            recall_succ+=1
        elif corrected_sent[0]==ins['target']:
            recall_err+=1

        if corrected_sent[0]==ins['target']:
            before_succ+=1
        else:
            before_err+=1
    print(corrected_new)
    submit.append({
        "inference": corrected_new,
        "id": ins['id']
    })
    idx += 1
print("loss_count, uncorrect, pred_succ, recall_succ, recall_err:",loss_count,uncorrect,pred_succ,recall_succ,recall_err)
print("before_succ, before_err",before_succ,before_err)
# json.dump(submit, open('../data/preliminary_a_data/output/preliminary_a_test_source.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

