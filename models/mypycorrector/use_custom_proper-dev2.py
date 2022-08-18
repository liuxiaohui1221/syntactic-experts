# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 自定义成语纠错
"""
import json
import os.path
import sys
from difflib import SequenceMatcher
from operator import itemgetter

from tqdm import tqdm

from ProjectPath import get_project_path
from knowledgebase.dict.check_proper import readWordFile, unique_file
from knowledgebase.tencent.SentenceSimilarity import WordSentenceSimliarity
from models.macbert.util.common import chooseBestCorrectCandidate
from models.mypycorrector.utils.text_utils import is_chinese

sys.path.append("..")
from models.mypycorrector.corrector import Corrector

def getEdits(src_text, trg_text):
    r = SequenceMatcher(None, src_text, trg_text)
    diffs = r.get_opcodes()
    edits = []
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        if "replace" !=tag:
            continue
        edits.append((src_text[i1:i2],trg_text[j1:j2]))
    return edits



if __name__ == '__main__':
    error_sentences = [
        "#真正的相濡以沫是怎样的#爱是罗曼蒂克，爱是细水流长，爱也是柴米油盐。",
        "疫情不反弹",
        "最后，由衷感谢您的辛勤耕耘、辛苦付出、育人育心，由衷祝您健康快乐、阖家幸福、桃李天下！"
        "确保防控不松懈、疫情不反殚",
        "而该车辆荷载人数仅7人",
        "青岛市人民检查院指控，1999年至2020年，董宏先后担任海南省委副秘书长、北京市政府副秘书长、中央巡视组副组长等职务，为他人非法谋利，收受财物4.6亿余元。",
        "以城为镜，弘扬中华民主传统美德，让我们一起用实际行动为信阳的文明发展贡献出自己的一份力量！",
    "警方当下立断，让其父亲将梯子架在屋檐旁，并在梯子底下埋伏一个突击小组。",
    "2021年8月9日傍晚6时左右，虹桥镇万源新城3期117号门口的非机动车车棚里一辆电动自行车充电时突然自燃，车内的电瓶发生爆炸，火势一簇而上，又因当天风力强劲，火势迅速向西蔓延。",
    "自禄口机场暴发疫情以来，潘金海积极响应号召，招募志愿者，成立美年大健康志愿者服务小分队，组织防疫培训、帮忙搭建遮阳棚、协助核酸采集……每天忙得马不停歇。",
    "走进县直机关入党积极分子暨发展对象培训班，和瑞庭同志用自己的亲生经历为这92名入党积极分子上了一堂深刻的爱国主义教育课。"
    ]
    # m = Corrector(proper_name_path='')
    # for i in error_sentences:
    #     print(i, ' -> ', m.correct(i))
    # text = '预计：明天夜里到6号白天，多云到阴，部分地区有分散型阵雨或雷雨。'
    print('*' * 42)
    proper_path=os.path.join(get_project_path(),'knowledgebase/dict/custom_dict.txt')
    confusion_path = os.path.join(get_project_path(), 'knowledgebase/confusion/good_confusions.txt')
    m = Corrector(custom_confusion_path=confusion_path,custom_word_freq_path=proper_path,proper_name_path=proper_path)
    # m = Corrector(custom_word_freq_path=proper_path,proper_name_path=proper_path)
    for i in error_sentences:
        print(i, ' -> ', m.correct(i, only_proper=True))

    error_sentences = json.load(
        open(os.path.join(get_project_path(), 'models/model_MiduCTC/data/preliminary_a_data/preliminary_val.json'),
             encoding='utf-8'))

    success,uncorrected,fail=0,0,0
    contains_num,total=0,0
    unchecked, recall_succ,recall_fail = 0, 0,0
    outPath='knowledgebase/dict/low_chengyu_dev.txt'
    maybe_bad_words=[]
    threshold = 0.015
    # tencet word2vec
    wss = WordSentenceSimliarity()
    for ins in error_sentences[:]:
        text=ins['source']
        target=ins['target']
        # 拼写纠错
        # text='在舒适性方面，云南省对路面破损、平整度、车辙三项指标有一项或多项达不到“优”的22条共2940.866公里单幅高速公路进行了集中处治。'
        # text='激情高涨，深情并茂的用朗诵的方式表达自己对中华文化'
        corrected = m.correct(text, only_proper=True,recall=True,exclude_proper=False,min_word_length=2,max_word_length=4)
        final_text=corrected[0]
        final_detail=corrected[1]
        recalled=False
        if len(corrected[1]) > 0:
            final_text,final_detail,recalled=chooseBestCorrectCandidate(wss,text,corrected[1],target,threshold=threshold)

            print(final_text,final_detail,recalled)
            if len(final_text)==0:
                final_text=corrected[0]
            else:
                final_text=final_text[0]

            if final_text==target or recalled:
                recall_succ+=1
            else:
                recall_fail+=1

        if final_text == target:
            success += 1
            print("Success correct:", final_text,final_detail,recalled)
        else:
            fail += 1
            print("Failed correct:", len(corrected[1]),final_text)
            print("Actual edits:",getEdits(text,target))
            maybe_bad_words.extend(corrected[1])
        if corrected[0] == text:
            unchecked += 1
        if len(maybe_bad_words)%50==0 and len(maybe_bad_words)>0:
            with open(os.path.join(get_project_path(), outPath), 'a+', encoding='utf-8') as f:
                for pair in maybe_bad_words:
                    f.write(pair[0] + '\t' + pair[1] + '\n')
            maybe_bad_words=[]
    print("Recall succ,fail:",recall_succ,recall_fail)
    print("Last succ,fail:",success,fail)
    if len(maybe_bad_words) > 0:
        with open(os.path.join(get_project_path(), outPath), 'w', encoding='utf-8') as f:
            for pair in maybe_bad_words:
                f.write(pair[0] + '\t' + pair[1] + '\n')
        maybe_bad_words = []
    print("low propers saved:",outPath)
    # 去重
    unique_file(outPath)

