# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 用户自定义混淆集，功能：1）补充纠错对，提升召回率；2）对误杀加白，提升准确率
"""
import json
import os.path
import sys

from tqdm import tqdm

from ProjectPath import get_project_path
from models.mypycorrector.corrector import Corrector

sys.path.append("..")
if __name__ == '__main__':
    error_sentences = [
        "1.朝天邑的苍穹岩石2.今天的一杯绿茶洞窟咖啡馆3.静石航空馆附近的森林隧道4.松堂里秘密森林期待疫情早日結束，大家一起加油喔！",
        "就是以联合国宪章宗旨和原则为基础的国际关系基本准则",
        '买iphonex，要多少钱',  # 漏召回
        '我想喝小明同学。',  # 漏召回
        '哪里卖苹果吧？请大叔给我让坐',  # 漏召回
        '交通先行了怎么过去啊？',  # 漏召回
        '共同实际控制人萧华、霍荣铨、张旗康',  # 误杀
        '上述承诺内容系本人真实意思表示',  # 正常
        '大家一哄而伞怎么回事',  # 成语
    ]
    # m = Corrector()
    # for i in error_sentences:
    #     print(i, ' -> ', m.correct(i))

    print('*' * 42)
    # confusion_path = os.path.join(get_project_path(), 'knowledgebase/confusion/good_confusions.txt')
    confusion_path = os.path.join(get_project_path(), 'knowledgebase/confusion/confusion_pair.txt')
    word_path = os.path.join(get_project_path(), 'knowledgebase/dict/custom_dict.txt')
    # m = Corrector(custom_confusion_path=confusion_path,word_freq_path=word_path,proper_name_path=word_path)
    m = Corrector(word_freq_path=word_path,proper_name_path=word_path)
    for i in error_sentences:
        print(i, ' -> ', m.correct(i))

    # test_data='preliminary_extend_train.json'
    test_data='preliminary_val.json'
    # test_data='final_val.json'
    success, uncorrected, fail = 0, 0, 0
    unchecked, total = 0, 0
    maybe_bad_words = []
    error_sentences = json.load(
        open(os.path.join(get_project_path(), f'models/model_MiduCTC/data/preliminary_a_data/{test_data}'),
             encoding='utf-8'))
    for ins in error_sentences[:]:
        text = ins['source']
        total += 1
        # for text in tqdm(error_sentences[:]):
        # text='在舒适性方面，云南省对路面破损、平整度、车辙三项指标有一项或多项达不到“优”的22条共2940.866公里单幅高速公路进行了集中处治。'
        # text='激情高涨，深情并茂的用朗诵的方式表达自己对中华文化'
        corrected = m.correct(text,only_proper=True,recall=False,exclude_proper=True,min_word_length=4,shape_score=0.85)
        if len(corrected[1]) > 0:
            if corrected[0] == ins['target']:
                success += 1
                # print("Success correct:", corrected)
            else:
                fail += 1
                maybe_bad_words.append(corrected[1])
                print("Failed correct:", corrected)
        if corrected[0] == ins['source']:
            unchecked += 1
        # print(corrected)
    print("total,success,fail,unchecked:", total, success, fail, unchecked)
