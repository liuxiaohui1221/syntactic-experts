# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 自定义成语纠错
"""
import json
import os.path
import sys
from difflib import SequenceMatcher

import jieba
from tqdm import tqdm

from ProjectPath import get_project_path

sys.path.append("..")
from models.mypycorrector.corrector import Corrector

def containKeyEdits(reference_list,src_text, trg_text):
    r = SequenceMatcher(None, src_text, trg_text)
    diffs = r.get_opcodes()
    m1_edits = []
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        if "equal" in tag:
            continue
        if "replace" !=tag:
            return False
        flag=False
        for ref_tuple in reference_list:
            if ref_tuple[0]==src_text[i1:i2] and ref_tuple[1]==trg_text[j1:j2]:
                flag=True
        if flag==False:
            return False
    return True

if __name__ == '__main__':
    error_sentences = [
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
    m = Corrector(custom_word_freq_path=proper_path,proper_name_path=proper_path,min_proper_len=4)
    for i in error_sentences:
        print(i, ' -> ', m.correct(i, only_proper=True))

    error_sentences = json.load(
        open(os.path.join(get_project_path(), 'models/model_MiduCTC/data/preliminary_a_data/preliminary_train.json'),
             encoding='utf-8'))

    success,uncorrected,fail=0,0,0
    contains_num,total=0,0
    unchecked, total = 0, 0
    maybe_bad_words=[]
    outPath='knowledgebase/dict/maybe_badword_dict.txt'
    for ins in tqdm(error_sentences[:]):
        text=ins['target']
        target=ins['target']
        # text='在舒适性方面，云南省对路面破损、平整度、车辙三项指标有一项或多项达不到“优”的22条共2940.866公里单幅高速公路进行了集中处治。'
        # text='激情高涨，深情并茂的用朗诵的方式表达自己对中华文化'
        corrected = m.correct(text, only_proper=True)
        if len(corrected[1]) > 0:
            if corrected[0] == target:
                success += 1
                print("Success correct:", corrected)
            else:
                fail += 1
                print("Failed correct:", corrected)
                maybe_bad_words.append(corrected[1])
        if corrected[0] == text:
            unchecked += 1
        if len(maybe_bad_words)%50==0 and len(maybe_bad_words)>0:
            with open(os.path.join(get_project_path(), outPath), 'a+', encoding='utf-8') as f:
                for pair in maybe_bad_words:
                    f.write(pair[0][0] + '\t' + pair[0][1] + '\n')
            maybe_bad_words=[]
    print(success,fail,total,contains_num)
    if len(maybe_bad_words) > 0:
        with open(os.path.join(get_project_path(), outPath), 'a+', encoding='utf-8') as f:
            for pair in maybe_bad_words:
                f.write(pair[0][0] + '\t' + pair[0][1] + '\n')
        maybe_bad_words = []

