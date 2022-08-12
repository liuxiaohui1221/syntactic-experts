# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os.path

from pycorrector.utils.eval import get_bcmi_corpus

from ProjectPath import get_project_path

data_path = os.path.join(get_project_path(),'knowledgebase/data/confusion/confusion_word.txt')
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        # 甘败(拜)下风 => 甘败下风	甘拜下风
        error_sentence, right_sentence, right_detail = get_bcmi_corpus(line, left_symbol='(', right_symbol=')')
        if not error_sentence:
            error_sentence, right_sentence, right_detail = get_bcmi_corpus(line, left_symbol='（', right_symbol='）')
            if not error_sentence:
                continue
        # print(right_detail)
        print(error_sentence + '\t' + right_sentence)
