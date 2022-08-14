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
from models.model_MiduCTC.src import corrector, correctorV3
from tqdm import tqdm
import json
import numpy
from models.model_MiduCTC.src import corrector
import argparse

from models.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf
from models.mypycorrector.corrector import Corrector

from models.mypycorrector.utils.text_utils import is_chinese

# testa_data = json.load(open(os.path.join(get_ecspell_path(),'Results/results/checkpoint-preliminary_a_test_source.json'),encoding='utf-8'))
# testa_data = json.load(open(os.path.join(get_project_path(),'models/model_MiduCTC/data/preliminary_a_data/preliminary_val.json'),encoding='utf-8'))
# testa_data = json.load(open(os.path.join(get_ecspell_path(),'Results/results/checkpoint-preliminary_extend_train.json'),encoding='utf-8'))
testa_data = json.load(open(os.path.join(get_ecspell_path(),'Results/results/checkpoint-preliminary_val.json'),encoding='utf-8'))

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
    # m2,m4:macbert,ecspell优先
    if m2_edits==m4_edits:
        return 2
    elif m3_edits==m4_edits:
        return 4
    else:
        if m3_edits==m2_edits:
            return 2
        else:
            # 优先选同音纠错
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
        # 词向量检测
        return m1_text
    else:
        # 拼写检测问题识别模型集成：
        # 1.少数服从多数
        # 2.同音优于跨音：ECSpell同音优于不纠？
        # 3.少纠优于多纠
        tar_edits = getTwoTextEdits(ins['source'], ins.get('target'))

        m2_macbert_edits = getTwoTextEdits(ins['source'], m2_text)
        m3_py_edits = getTwoTextEdits(ins['source'], corrected_sent4)
        m4_ecspell_edits = getTwoTextEdits(ins['source'], ins['ecspell'])

        # 方式1：少数服从多数（先4个，没有多数时：macbert与ecspell选同音字纠少的）
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
        # 词向量检测
        return m1_text
    else:
        # 拼写检测问题
        if m1_text!=ins['ecspell'] and m1_text==ins['source']:
            # m1模型漏检
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
        # 词向量检测
        return m1_text
    else:
        # 拼写检测问题
        if m1_text!=ins['ecspell'] and m1_text==ins['source']:
            # m1模型漏检
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
    # 优先检测集合
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


def stopDuplicateCheck(w,word):
    stopchecks=['队','军','每','图片','妈妈','由','村','市','丝']
    stopwords=['不着急','丝丝']
    if w in stopchecks or word in stopwords:
        return True
    return False


def removeDuplicate(fenci, text):
    arr=fenci.lcut(text)
    # 相邻存在包含关系的
    pre=None
    pre_index=-1
    fine_text = numpy.array(arr)
    for index,word in enumerate(arr):
        fine_text[index] = word
        flag=False
        for w in word:
            if is_chinese(w)==False and stopDuplicateCheck(w,word):
                flag=True
                break
        if flag==False:
            if pre and (len(pre)>1 or len(word)>1):
                if pre and len(word)>len(pre) and word[:len(pre)]==pre:
                    # del pre
                    print("del word:", fine_text[pre_index],"from:",text)
                    fine_text[pre_index]=''
                elif pre and len(pre)>=len(word) and pre[len(pre)-len(word):]==word:
                    # del cur word
                    print("del word:",fine_text[index],"from:",text)
                    fine_text[index]=''
        pre=word
        pre_index=index
    return "".join(fine_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--macbert_model_dir", default='pretrained/macbert4csc',
                        type=str,
                        help="MacBert pre-trained model dir")
    args = parser.parse_args()
    fenci=VocabConf().jieba_singleton
    # 模型
    ctc_correct = corrector.Corrector(
        os.path.join(get_project_path(),
                     'models/model_MiduCTC/model/epoch3,step1,testf1_62_93%,devf1_47_38%')
        , ctc_label_vocab_dir=os.path.join(get_project_path(), 'models/model_MiduCTC/src/baseline/ctc_vocab'))
    m = MacBertCorrector(args.macbert_model_dir)
    # proper_path = os.path.join(get_project_path(), 'knowledgebase/dict/chengyu.txt')
    confusion_path = os.path.join(get_project_path(), 'models/mypycorrector/data/confusion_pair.txt')
    word_path = os.path.join(get_project_path(), 'knowledgebase/dict/custom_dict.txt')
    m4 = Corrector(custom_confusion_path=confusion_path, word_freq_path=word_path, proper_name_path=word_path,
                  min_proper_len=4)

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
        # 去除重复词
        src_text = removeDuplicate(fenci, ins['source'])
        if src_text==ins['source']:
            # ins['source']=src_text
            # 比较拼写纠错问题
            corrected_sent = ctc_correct([src_text])

            corrected_sent2 = m.macbert_correct(src_text)
            # corrected_sent3 = m.macbert_correct_recall(src_text,val_target=ins.get('target'))

            corrected_sent4, detail = m4.correct(src_text, only_proper=True)
            # 判断是否为拼写纠错: m2纠错字为音近形近字（m1预测为非拼写问题时使用，否则按长度比较）
            final_corrected=predictAgainM1M2Tenc(corrected_sent[0],corrected_sent2[0],ins)
        # final_corrected2 = predictAgain(corrected_sent[0], corrected_sent3[0], corrected_sent4, ins, score_compares_recall_in_spell,
        #                                 fieldnames,scores=corrected_sent3[1],first_correct=corrected_sent3[2])
        else:
            corrected_sent4=src_text
            corrected_sent=src_text
            corrected_sent2=src_text
            final_corrected=src_text
            detail=""

        final_corrected2=""
        m1_edits = getTwoTextEdits(src_text, corrected_sent[0])
        m2_edits = getTwoTextEdits(src_text, corrected_sent2[0])
        m1m2_edits = getTwoTextEdits(src_text, final_corrected)
        m1m2_recall_edits = getTwoTextEdits(src_text, final_corrected2)
        ecspell_edits = getTwoTextEdits(src_text, ins['ecspell'])
        # 检测集
        # candidate_check_words=getCandidateCheckWords(m1_edits,m2_edits,m1m2_edits,m1m2_recall_edits,ecspell_edits)


        finale_corrected3 = predictAgainM1M2PyDict(corrected_sent[0],corrected_sent2[0],ins,corrected_sent4)
        # if corrected_sent[0]!=ins['target'] and corrected_sent2[0]!=ins['target'] \
        #         and corrected_sent3[0]!=ins['target']  and corrected_sent4==ins['target']:
        #     pyc_right+=1
        if corrected_sent4!=ins['source']:
            # if corrected_sent4!=ins['target']:
            #     print("pydict correct:",corrected_sent4,detail)
            final_corrected=corrected_sent4
        submit.append({
            "inference": final_corrected,
            "id": ins['id']
        })
        if final_corrected!=final_corrected2:
            diff_correct+=1
        tar_edits = getTwoTextEdits(src_text, ins['target'])

        m4_edits = getTwoTextEdits(src_text, corrected_sent4)
        diff_correct_results.append({
            diff_names[0]:corrected_sent[0]==ins['target'],
            diff_names[1]: corrected_sent2[0] == ins['target'],
            diff_names[2]:final_corrected==ins['target'],
            diff_names[3]: final_corrected2 == ins['target'],
            diff_names[4]: ins['ecspell_flag'],
            diff_names[5]: corrected_sent4 == ins['target'],
            diff_names[6]:tar_edits,
            diff_names[7]:m1_edits,
            diff_names[8]: m2_edits,
            diff_names[9]: m1m2_edits,
            diff_names[10]:m1m2_recall_edits,
            diff_names[11]: ecspell_edits,
            diff_names[12]: m4_edits,
            diff_names[13]:"",
            diff_names[14]: final_corrected2,
            diff_names[15]: ins['source'],
            diff_names[16]: ins['target'],
            diff_names[17]:ins['type']
        })
        if ins['source']==ins['target']:
            pos_nums+=1
        else:
            neg_nums+=1
        if final_corrected==ins['target']:
            s1s2+=1
        if ins.get('target')==ins.get('ecspell'):
            ecspell+=1
        # if finale_corrected3==ins['target']:
        #     s1s2_pydict+=1
        if corrected_sent4==ins['target']:
            pydict+=1
        if final_corrected2==ins['target']:
            s1s2_recall+=1
            if len(ins['source'])==len(ins['target']):
                s1s2_recall_spell+=1
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
            spellNums+=1
            if corrected_sent[0] == corrected_sent2[0]:
                commons+=1
                if corrected_sent2[0]==ins['target']:
                    right_comm+=1

            if corrected_sent[0] == ins['target']:
                s1_in_spell+=1
            if corrected_sent[0] == ins['source'] and ins['source'] != ins['target']:
                m1_lou_jian += 1

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
        idx += 1
    print(equ_nums,idx)
    print("exceed,total nums,pos_nums,neg_nums:",exceed_max,idx,pos_nums,neg_nums)
    print("All: s1,s2,s1s2,s1s2s3s4,ecspell,s1s2_pydict,pydict,check_no_spell:",s1,s2,s1s2,s1s2_recall,ecspell,s1s2_pydict,pydict,check_no_spell)
    print("Nospell s1,s2:",s1_nospell,s2_nospell)
    print("Spell nums,s1,s2,s1s2_spell,s1s2_recall_spell:",spellNums,s1_in_spell,s2_in_spell,s1s2_spell,s1s2_recall_spell)
    print("Spell m1_lou_jian,right_comm,s2_in_m1_err,s2_in_m1_ignore,s2_in_m1_nospells:",m1_lou_jian,right_comm,s2_in_m1,s2_in_m1_ignore,s2_in_m1_predict_nospells)
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

    saveCSV(diff_correct_results, "./output/preliminary_val_compare_recall.csv", diff_names)

    saveCSV(score_compares_in_spell,"./output/preliminary_val_compare_score_spell.csv",fieldnames)

    saveCSV(score_compares_recall_in_spell,"./output/preliminary_val_compare_score_recall_spell.csv",fieldnames)



