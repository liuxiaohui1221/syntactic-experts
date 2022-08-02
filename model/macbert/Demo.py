import os.path
from difflib import SequenceMatcher

import torch
from tqdm import tqdm
import json

from ProjectPath import get_project_path
from model.macbert.macbert_corrector import MacBertCorrector
from model.model_MiduCTC.src import corrector, correctorV3
from tqdm import tqdm
import json
import numpy
from model.model_MiduCTC.src import corrector
import argparse
# testa_data = json.load(open(os.path.join(get_project_path(),'model/model_MiduCTC/data/preliminary_a_data/preliminary_a_test_source.json'),encoding='utf-8'))
testa_data = json.load(open(os.path.join(get_project_path(),'model/model_MiduCTC/data/preliminary_a_data/preliminary_val.json'),encoding='utf-8'))
# 模型
# correct= corrector.Corrector('../new_model/ctc_2022Y07M21D01H/epoch2,step62500,testf1_35_77%,devf1_35_77%')
# correct= corrector.Corrector('../new_model/ctc_2022Y07M21D08H/epoch4,step1,testf1_35_94%,devf1_35_94%')
correct= corrector.Corrector(os.path.join(get_project_path(),'model/model_MiduCTC/model/epoch3,step1,testf1_62_93%,devf1_47_38%')
                             ,ctc_label_vocab_dir=os.path.join(get_project_path(),'model/model_MiduCTC/src/baseline/ctc_vocab'))


def predictAgain(m1_text, m2_text, ins):
    if len(m1_text)!=len(ins['source']):
        if len(m2_text)==len(ins['source']):
            # todo 混淆集判断是否为拼写问题
            pass
        else:
            return m1_text
    elif len(m2_text)!=len(ins['source']):
        # 词向量检测
        return m1_text
    else:
        # 拼写检测问题
        if m1_text!=m2_text and m1_text==ins['source']:
            # m1模型漏检
            # todo 词向量继续检测
            # if m1_text == ins['target']:
            #     print("1.",m2_text,ins['target'],m1_text==ins['target'])
            # isReplace,score = doReplace(ins['source'], m2_text)
            # print(isReplace,score)
            # if isReplace:
            #     return m2_text
            # else:
            #     return ins['source']
            return m2_text
        elif m1_text!=m2_text and m2_text==ins['source']:
            # todo m2漏检，词向量继续检测
            # if m2_text == ins['target']:
            #     print("2.", m2_text == ins['target'])
            # isReplace, score = doReplace(ins['source'], m1_text)
            # if isReplace:
            #     return m1_text
            # else:
            #     return ins['source']

            return m1_text
        elif m1_text!=m2_text:
            # todo 拼写纠错不一致，词向量继续检测
            # if m1_text == ins['target']:
            #     print("3.", m2_text,ins['source'],ins['target'],m1_text == ins['target'])
            # isReplace1, score1 = doReplace(ins['source'], m1_text)
            # isReplace2, score2 = doReplace(ins['source'], m2_text)
            # print(isReplace1,score1,isReplace2,score2)
            # if score2>score1:
            #     return m2_text
            # else:
            #     return m1_text

            return m2_text
    return m1_text
def ignoreNonChineseCorrect(source,reference):
    # 模型检错纠错忽略英文字母大小写纠错，以及UNK标识
    r = SequenceMatcher(None, source, reference)
    diffs = r.get_opcodes()
    wrongIds = []

    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        if tag == 'replace':
            wrongIds.extend(range(i1, i2))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--macbert_model_dir", default='pretrained/macbert4csc',
                        type=str,
                        help="MacBert pre-trained model dir")
    args = parser.parse_args()

    nlp = MacBertCorrector(args.macbert_model_dir).macbert_correct
    submit = []
    idx=0
    equ_nums=0
    s1_in_spell,s2_in_spell,commons=0,0,0
    exceed_max=0
    diff=[]
    not_check=0
    check_no_spell=0
    nospells=[]
    s1_nospell,s2_nospell=0,0
    success_com=0
    right_comm=0
    s2_in_m1=0
    s2_in_m1_ignore,s2_in_m1_predict_nospells=0,0
    whatserror_in_m1=[]
    m2_err_in_m1=0
    comon_errs=[]
    s1,s2,s1s2=0,0,0
    m2_errs_in_pos_m1_right=[]
    m2_predict_nospells,m2_predict_nospells_right,m2_predict_actual_nospell=0,0,0
    m2_predict_to_nospells_in_spell=0
    m1_predict_right_in_m2_pred_nospell=0
    s2_in_m1_nospells=[]
    s1_or_s2=0
    for ins in tqdm(testa_data[:]):
        # 比较拼写纠错问题
        corrected_sent = correct(ins['source'])
        corrected_sent2 = nlp(ins['source'])
        # 判断是否为拼写纠错: m2纠错字为音近形近字（m1预测为非拼写问题时使用，否则按长度比较）
        final_corrected=predictAgain(corrected_sent[0],corrected_sent2[0],ins)
        submit.append({
            "inference": final_corrected,
            "id": ins['id']
        })
        if final_corrected==ins['target']:
            s1s2+=1
        # corrected_sent2 = nlp_macbert(ins['source'])
        if corrected_sent[0]==ins['target']:
            s1+=1
        if corrected_sent2[0] == ins['target']:
            s2 += 1
        # 融合：若m1检测出是非拼写问题或者m2检测为非拼写问题，则使用m1的预测，否则，若两者均纠错了拼写问题且不等，或者只有一个存在纠错，则使用腾讯词向量
        if len(corrected_sent2[0])!=len(ins['source']):
            m2_predict_nospells+=1
            if len(ins['source'])!=len(ins['target']):
                m2_predict_actual_nospell+=1
            else:
                m2_predict_to_nospells_in_spell+=1
            if corrected_sent2[0]==ins['target']:
                m2_predict_nospells_right+=1
            if corrected_sent[0] == ins['target']:
                m1_predict_right_in_m2_pred_nospell += 1
        if len(ins['source'])!=len(ins['target']):
            if corrected_sent2[0]==ins['source']:
                not_check+=1
            else:
                # print(corrected_sent2[0],ins['source'],ins['target'])
                check_no_spell+=1
                nospells.append({
                    "source": ins['source'],
                    "target": ins['target'],
                    "type": ins['type'],
                    "inference1": corrected_sent[0],
                    "inference2": corrected_sent2[0]
                })
            if  corrected_sent[0]==ins['target']:
                s1_nospell+=1
            if corrected_sent2[0] == ins['target']:
                s2_nospell += 1

        else:
            if corrected_sent[0] == corrected_sent2[0]:
                commons+=1
                if corrected_sent2[0]==ins['target']:
                    right_comm+=1

            if corrected_sent[0] == ins['target']:
                s1_in_spell+=1

            if corrected_sent2[0]==ins['target']:
                s2_in_spell+=1
            if corrected_sent[0] == ins['target'] or corrected_sent2[0]==ins['target']:
                s1_or_s2+=1
            if corrected_sent[0]!=ins['target']:
                if corrected_sent2[0]==ins['target']:
                    s2_in_m1+=1
                    whatserror_in_m1.append({
                        "source": ins['source'],
                        "target": ins['target'],
                        "type": ins['type'],
                        "inference1": corrected_sent[0],
                        "inference2": corrected_sent2[0]
                    })
                    if ins['source']==corrected_sent[0]:
                        # m1漏检
                        s2_in_m1_ignore+=1
                    if len(ins['source'])!=len(corrected_sent[0]):
                        s2_in_m1_predict_nospells+=1
                        s2_in_m1_nospells.append({
                            "source": ins['source'],
                            "target": ins['target'],
                            "type": ins['type'],
                            "inference1": corrected_sent[0],
                            "inference2": corrected_sent2[0]
                        })
                else:
                    # common error
                    comon_errs.append({
                        "source": ins['source'],
                        "target": ins['target'],
                        "type":ins['type'],
                        "inference1": corrected_sent[0],
                        "inference2": corrected_sent2[0]
                    })
            elif ins['source']==corrected_sent[0]:
                # 实为正，m2误纠
                if corrected_sent2[0]!=ins['target']:
                    m2_err_in_m1+=1
                    m2_errs_in_pos_m1_right.append({
                        "source": ins['source'],
                        "target": ins['target'],
                        "type": ins['type'],
                        "inference1": corrected_sent[0],
                        "inference2": corrected_sent2[0]
                    })
            if corrected_sent[0] == ins['target'] and corrected_sent2[0]==ins['target']:
                success_com+=1
            submit.append({
                "source": ins['source'],
                "target": ins['target'],
                "type":ins['type'],
                "inference1": corrected_sent[0],
                "inference2": corrected_sent2[0]
            })
        idx += 1
    print(submit,equ_nums,idx)
    print("exceed,total nums:",exceed_max,idx)
    print("All: s1,s2,s1s2,commons,not_check,check_no_spell:",s1,s2,s1s2,commons,not_check,check_no_spell)
    print("Nospell s1,s2:",s1_nospell,s2_nospell)
    print("Spell s1,s2:",s1_in_spell,s2_in_spell)
    print("Spell nums,right_comm,s2_in_m1_err,s2_in_m1_ignore,s2_in_m1_nospells:",len(submit),right_comm,s2_in_m1,s2_in_m1_ignore,s2_in_m1_predict_nospells)
    print("Spell 得分=+s2_in_m1_ignore,-m2_err_in_m1:",s2_in_m1_ignore,m2_err_in_m1)
    print("Spell m2 predict to nospells,m2_predict_nospells_right,m1_predict_right_in_m2_pred_nospell,actualspell:",
          m2_predict_nospells,m2_predict_nospells_right,m1_predict_right_in_m2_pred_nospell,m2_predict_actual_nospell)
    print("Spell m2 predict to nospells in spell,s1_or_s2:",m2_predict_to_nospells_in_spell,s1_or_s2)
    # json.dump(submit, open('./output/preliminary_a_test_source.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    json.dump(nospells, open('./output/preliminary_val_compare_nospell_corrected.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    json.dump(whatserror_in_m1, open('./output/preliminary_val_compare_whatserror_in_m1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    json.dump(comon_errs, open('./output/preliminary_val_compare_comon_errs.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    json.dump(m2_errs_in_pos_m1_right, open('./output/preliminary_val_compare_m2_errs_in_pos_m1_right.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    json.dump(s2_in_m1_nospells,
              open('./output/preliminary_val_compare_s2_in_m1_nospells.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)


