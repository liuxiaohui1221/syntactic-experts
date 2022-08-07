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
        '这块民表带带相传',
        '他贰话不说把牛奶喝完了',
        '消炎可以吃点阿木西林药品',  # 阿莫西林
    ]
    # m = Corrector(proper_name_path='')
    # for i in error_sentences:
    #     print(i, ' -> ', m.correct(i))
    text = '预计：明天夜里到6号白天，多云到阴，部分地区有分散型阵雨或雷雨。'
    print('*' * 42)
    proper_path=os.path.join(get_project_path(),'knowledgebase/dict/custom_dict.txt')
    m = Corrector(custom_word_freq_path=proper_path,proper_name_path=proper_path)
    testa_data = json.load(
        open(os.path.join(get_project_path(), 'model/model_MiduCTC/data/preliminary_a_data/preliminary_val.json'),
             encoding='utf-8'))
    success,uncorrected,fail=0,0,0
    contains_num,total=0,0
    for ins in tqdm(testa_data[:]):
        if len(ins['source'])!=len(ins['target']):
            continue
        # text='在舒适性方面，云南省对路面破损、平整度、车辙三项指标有一项或多项达不到“优”的22条共2940.866公里单幅高速公路进行了集中处治。'
        # text='激情高涨，深情并茂的用朗诵的方式表达自己对中华文化'
        corrected=m.correct(ins['source'])
        if corrected[0]==ins['target']:
            success+=1
        elif ins['source']!=ins['target']:
            fail+=1
            # 判断是否召回
            if corrected[0]==ins['source']:
                isrecalled=False
            else:
                isrecalled=containKeyEdits(corrected[1],ins['source'],ins['target'])
            if isrecalled:
                contains_num+=1
            print("source,reference,contains:",ins['source'],len(corrected[1]),isrecalled)
        total+=1
        if corrected[0]==ins['source']:
            uncorrected+=1
    print(success,fail,total,contains_num)
    seg_list = jieba.cut(text, cut_all=False)  # 使用精确模式进行分词
    print("/".join(seg_list))
