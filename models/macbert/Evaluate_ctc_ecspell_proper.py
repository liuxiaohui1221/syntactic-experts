import csv
import os.path
from difflib import SequenceMatcher

import pycorrector
import torch
from tqdm import tqdm
import json

from ProjectPath import get_project_path
from knowledgebase.chinese_pinyin_util import ChinesePinyinUtil
from knowledgebase.tencent.SentenceSimilarity import WordSentenceSimliarity
from models.ECSpell.Code.ProjectPath import get_ecspell_path
from models.macbert.macbert_corrector import MacBertCorrector
from models.macbert.util.common import removeDuplicate
from models.model_MiduCTC.src import corrector, correctorV3
from tqdm import tqdm
import json
import numpy
from models.model_MiduCTC.src import corrector
import argparse

from models.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf
from models.mypycorrector.corrector import Corrector

from models.mypycorrector.utils.text_utils import is_chinese

testa_data = json.load(open(os.path.join(get_ecspell_path(),'Code/Results/ecspell/results/checkpoint-preliminary_b_test_source.json'),encoding='utf-8'))
# testa_data = json.load(open(os.path.join(get_project_path(),'models/model_MiduCTC/data/preliminary_a_data/preliminary_val.json'),encoding='utf-8'))
# testa_data = json.load(open(os.path.join(get_ecspell_path(),'Code/Results/ecspell/results/checkpoint-preliminary_extend_train.json'),encoding='utf-8'))
# testa_data = json.load(open(os.path.join(get_ecspell_path(),'Code/Results/ecspell/results/checkpoint-preliminary_val.json'),encoding='utf-8'))

# wss=WordSentenceSimliarity()
pyUtil=ChinesePinyinUtil()
def getTwoTextEdits(src_text, m1_text):
    if m1_text==None:
        return None
    r = SequenceMatcher(None, src_text, m1_text)
    diffs = r.get_opcodes()
    m1_edits = []
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        if "equal" in tag:
            continue
        m1_edits.append((tag,ins['source'][i1:i2],m1_text[j1:j2]))
    return m1_edits
def getTwoTextEditsV2(src_text, m1_text):
    if m1_text==None:
        return None
    r = SequenceMatcher(None, src_text, m1_text)
    diffs = r.get_opcodes()
    m1_edits = []
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        if "equal" in tag:
            continue
        m1_edits.append(diff)
    return m1_edits

def checkSamePinyin(m2_edits, m3_edits, m4_edits):
    cpy_m2=isSamePyin(m2_edits)
    cpy_m3=isSamePyin(m3_edits)
    cpy_m4 = isSamePyin(m4_edits)
    if cpy_m2:
        return 2
    elif cpy_m3:
        return 3
    elif cpy_m4:
        return 4
    else:
        return 4

def isSamePyin(edits):
    corePy_m2 = []
    corePy_src = []
    for tuple in edits:
        for word in tuple[2]:
            corePy_m2.append(pyUtil.getCorePinyinByChinese(word))
        for word in tuple[1]:
            corePy_src.append(pyUtil.getCorePinyinByChinese(word))
    if len(corePy_m2)==0:
        return False
    return corePy_m2==corePy_src

def chooseMultiPredict(tar_edits,m2_edits, m3_edits, m4_edits):
    # m2,m4:macbert,ecspell??????
    if m2_edits==m4_edits:
        return 2
    elif m3_edits==m4_edits:
        return 4
    else:
        if m3_edits==m2_edits:
            return 2
        else:
            # ?????????????????????
            model_num=checkSamePinyin(m2_edits,m3_edits,m4_edits)
            # print("All predict diff", m2_edits, m3_edits, m4_edits,"tar_edits:",tar_edits,"commit model:",model_num)
            return model_num

def predictAgain(m1_text, m2_text, corrected_sent4,ins,score_compares_in_spell,fieldnames,scores=None,first_correct=None):
    m1_edits = getTwoTextEdits(ins['source'], m1_text)
    if len(m1_text)!=len(ins['source']):
        for edit in m1_edits:
            if len(edit[1])>2:
                return m2_text
        return m1_text
    elif len(m2_text)!=len(ins['source']):
        # ???????????????
        return m1_text
    else:
        # ???????????????????????????????????????
        # 1.??????????????????
        # 2.?????????????????????ECSpell?????????????????????
        # 3.??????????????????
        tar_edits = getTwoTextEdits(ins['source'], ins.get('target'))

        m2_macbert_edits = getTwoTextEdits(ins['source'], m2_text)
        m3_py_edits = getTwoTextEdits(ins['source'], corrected_sent4)
        m4_ecspell_edits = getTwoTextEdits(ins['source'], ins['ecspell'])

        # ??????1???????????????????????????4????????????????????????macbert???ecspell????????????????????????
        model_num=chooseMultiPredict(tar_edits,m2_macbert_edits,m3_py_edits,m4_ecspell_edits)
        if model_num==1:
            return m1_text
        elif model_num==2:
            return m2_text
        elif model_num == 3:
            return corrected_sent4
        elif model_num==4:
            return ins['ecspell']
    return ins['ecspell']

def predictAgainM1M2Tenc(m1_text, m2_text, ins):
    if len(m1_text)!=len(ins['source']):
        return m1_text
    elif len(ins['ecspell'])!=len(ins['source']):
        # ???????????????
        return m1_text
    else:
        # ??????????????????
        if m1_text!=ins['ecspell'] and m1_text==ins['source']:
            # m1????????????
            return ins['ecspell']
        elif m1_text!=ins['ecspell'] and ins['ecspell']==ins['source']:
            return m1_text
        elif m1_text!=ins['ecspell']:
            return ins['ecspell']
    return ins['ecspell']


def existsEdit(i1, i2, m1_edits):
    flag=False
    for edit in m1_edits:
        tag, s_1, s_2, j1, j2 = edit
        if s_1==i1 and s_2==i2 and tag=='replace':
            flag=True
            break
    return flag


def findCommonDectect(src_text, pydict_text, m1_edits, m2_macbert_edits, m3_py_edits, m4_ecspell_edits):
    final_text=src_text
    for py_edit in m3_py_edits:
        tag, i1, i2, j1, j2 = py_edit
        if tag=="equal":
            continue
        flag1=existsEdit(i1,i2,m1_edits)
        flag2 = existsEdit(i1, i2, m2_macbert_edits)
        flag3 = existsEdit(i1, i2, m4_ecspell_edits)
        if flag1 or flag2 or flag3:
            final_text=final_text[:i1]+pydict_text[j1:j2]+final_text[i2:]
    return final_text

def predictAgainM1M2PyDict(m1_text, m2_text, ins, pydict_text):
    m1_edits = getTwoTextEditsV2(ins['source'], m1_text)
    m2_macbert_edits = getTwoTextEditsV2(ins['source'], m2_text)
    m3_py_edits = getTwoTextEditsV2(ins['source'], pydict_text)
    m4_ecspell_edits = getTwoTextEditsV2(ins['source'], ins['ecspell'])

    # common_dectect=findCommonDectect(ins['source'],pydict_text,m1_edits,m2_macbert_edits,m3_py_edits,m4_ecspell_edits)
    if pydict_text!=ins['source']:
        common_edit = getTwoTextEdits(ins['source'], pydict_text)

        print("pydict_text detect:",pydict_text,"common_edit :",common_edit)
        return pydict_text
    if len(m1_text)!=len(ins['source']):
        return m1_text
    elif len(ins['ecspell'])!=len(ins['source']):
        # ???????????????
        return m1_text
    else:
        # ??????????????????
        if m1_text!=ins['ecspell'] and m1_text==ins['source']:
            # m1????????????
            return ins['ecspell']
        elif m1_text!=ins['ecspell'] and ins['ecspell']==ins['source']:
            return m1_text
        elif m1_text!=ins['ecspell']:
            return ins['ecspell']
    return ins['ecspell']
def saveCSV(data_dicts,filepath,fieldnames):
    with open(filepath,"w",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        firstWrite=False
        for json_row in data_dicts:
            if firstWrite==False:
                writer.writeheader()
                firstWrite=True
            writer.writerow(json_row)


def getCandidateCheckWords(m1_edits, m2_edits, m1m2_edits, m1m2_recall_edits, ecspell_edits):
    # ??????????????????
    candidates=[]
    if len(m1_edits)>0:
        for edit in m1_edits:
            candidates.append(edit[1])
    if len(m2_edits)>0:
        for edit in m2_edits:
            candidates.append(edit[1])
    if len(m1m2_edits) > 0:
        for edit in m1m2_edits:
            candidates.append(edit[1])
    if len(m1m2_recall_edits) > 0:
        for edit in m1m2_recall_edits:
            candidates.append(edit[1])
    if len(ecspell_edits) > 0:
        for edit in ecspell_edits:
            candidates.append(edit[1])
    return set(candidates)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--macbert_model_dir", default='pretrained/macbert4csc',
                        type=str,
                        help="MacBert pre-trained model dir")
    args = parser.parse_args()
    fenci=VocabConf().jieba_singleton
    # ??????
    ctc_correct = corrector.Corrector(
        os.path.join(get_project_path(),
                     'models/model_MiduCTC/model/epoch3,step1,testf1_62_93%,devf1_47_38%')
        , ctc_label_vocab_dir=os.path.join(get_project_path(), 'models/model_MiduCTC/src/baseline/ctc_vocab'))
    m = MacBertCorrector(args.macbert_model_dir)
    # proper_path = os.path.join(get_project_path(), 'knowledgebase/dict/chengyu.txt')
    confusion_path = os.path.join(get_project_path(), 'models/mypycorrector/data/confusion_pair.txt')
    word_path = os.path.join(get_project_path(), 'knowledgebase/dict/custom_dict.txt')
    m4 = Corrector(custom_confusion_path=confusion_path, word_freq_path=word_path, proper_name_path=word_path)
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
    s1,s2,s1s2,s1s2_recall,s1s2_pydict,pydict=0,0,0,0,0,0
    m2_errs_in_pos_m1_right=[]
    m2_predict_nospells,m2_predict_nospells_right,m2_predict_actual_nospell=0,0,0
    m2_predict_to_nospells_in_spell=0
    m1_predict_right_in_m2_pred_nospell=0
    s2_in_m1_nospells,score_compares_in_spell,score_compares_recall_in_spell=[],[],[]
    s1_or_s2,spellNums,m1_lou_jian=0,0,0
    s1s2_spell,pos_nums,neg_nums,s1s2_recall_spell=0,0,0,0
    diff_correct,ecspell=0,0
    diff_correct_results=[]
    fieldnames = ["M1_score", "M2_score", "M1_interfer", "M2_interfer", "target_edits", "M1_edits","M2_edits","M2_first_edits","candidate_scores", "source", "target", "type"]

    diff_names=["M1","M2","M1M2","M1M2Recall_Py_Ecs","ECSpell","Py_dict","target_edits","M1_edits","M2_edits","M1M2_edits",
                "M1M2_Recall_eidts","ECSpell_edits","Pydict_eidts","correct2_scores","M1M2_recall_text","source","target","type"]
    pyc_right=0
    for ins in tqdm(testa_data[:]):
        # ???????????????
        src_text = removeDuplicate(fenci, ins['source'])
        if src_text==ins['source']:
            # ins['source']=src_text
            # ????????????????????????
            corrected_sent = ctc_correct([src_text])

            # corrected_sent2 = m.macbert_correct(src_text)
            # corrected_sent3 = m.macbert_correct_recall(src_text,val_target=ins.get('target'))
            corrected_sent4, detail = m4.correct(src_text, only_proper=True)
            # ???????????????????????????: m2??????????????????????????????m1????????????????????????????????????????????????????????????
            final_corrected=predictAgainM1M2Tenc(corrected_sent[0],None,ins)
        else:
            corrected_sent4=src_text
            corrected_sent=src_text
            corrected_sent2=src_text
            final_corrected=src_text
            detail=""

        if corrected_sent4!=ins['source']:
            final_corrected=corrected_sent4
        submit.append({
            "inference": final_corrected,
            "id": ins['id']
        })

    json.dump(submit, open('./output/preliminary_b_test_source.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print(equ_nums,idx)




