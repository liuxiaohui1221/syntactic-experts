# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 专名纠错，包括成语纠错、人名纠错、机构名纠错、领域词纠错等
"""
import json
import os
import random
from codecs import open

import pypinyin
from loguru import logger
from tqdm import tqdm

from models.mypycorrector import config
from knowledgebase.chinese_pinyin_util import ChinesePinyinUtil
from knowledgebase.tencent.SentenceSimilarity import WordSentenceSimliarity
from models.mypycorrector.utils.math_utils import edit_distance
from models.mypycorrector.utils.ngram_util import NgramUtil
from models.mypycorrector.utils.text_utils import is_chinese
from models.mypycorrector.utils.tokenizer import segment, split_2_short_text
from collections import defaultdict

def getNoTonePyin(pyUtil,p_tone):
    if pyUtil.isNumber(p_tone[-1]):
        return p_tone[:-1]
    return p_tone


def getDoubleKey(pyUtil,word_group):
    pysTone = pypinyin.lazy_pinyin(word_group, errors='ignore',style=pypinyin.Style.TONE3)
    corePysNoTone = []
    corePysTone = []
    for p_tone in pysTone:
        p_no_tone = getNoTonePyin(pyUtil, p_tone)
        cpyinNoTone = pyUtil.handleSimPinyinToCore(p_no_tone)
        cpyinTone = pyUtil.handleSimPinyinToCore(p_tone)
        corePysNoTone.append(cpyinNoTone)
        corePysTone.append(cpyinTone)

    key_core_no_tone = "_".join(corePysNoTone)
    key_core_tone = "_".join(corePysTone)
    return key_core_no_tone,key_core_tone

def getMappingProper(pyUtil,words,min_proper_len=3):
    # 格式：{core_pin_yin:{ pin_yin:[词组] }}
    corePyinDB=defaultdict(dict)
    for word_group in tqdm(words):
        if len(word_group)<min_proper_len:
            continue
        key_core_no_tone,key_core_tone=getDoubleKey(pyUtil,word_group)
        if key_core_no_tone in corePyinDB:
            pinyinToneDb = corePyinDB[key_core_no_tone]
            if key_core_tone in pinyinToneDb:
                pinyinToneDb[key_core_tone].append(word_group)
            else:
                pinyinToneDb[key_core_tone] = [word_group]
        else:
            pinyinToneDb={}
            pinyinToneDb[key_core_tone] = [word_group]
            corePyinDB[key_core_no_tone] = pinyinToneDb
    return corePyinDB


def load_set_file(pyUtil,path,min_proper_len=2):
    print("load proper file:","os.path.dirname(os.path.realpath(__file__))=%s" % os.path.dirname(path))
    proper_path=os.path.join(os.path.dirname(os.path.realpath(path)),str(min_proper_len)+"_gen_proper.txt")
    if os.path.exists(proper_path):
        corePyins=json.load(open(proper_path,encoding='utf-8'))
        return corePyins
    else:
        words = []
        if proper_path and os.path.exists(proper_path):
            with open(proper_path, 'r', encoding='utf-8') as f:
                for w in f:
                    w = w.strip()
                    if w.startswith('#'):
                        continue
                    if w:
                        words.append(w)
        # 转换并保存
        corePyins=getMappingProper(pyUtil,words,min_proper_len=2)
        json.dump(corePyins,open(proper_path, 'w', encoding='utf-8'),ensure_ascii=False, indent=4)
        print("Load proper file over!")
        return corePyins

def load_dict_file(path):
    """
    加载词典
    :param path:
    :return:
    """
    result = {}
    if path:
        if not os.path.exists(path):
            logger.warning('file not found.%s' % path)
            return result
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        continue
                    terms = line.split()
                    if len(terms) < 2:
                        continue
                    result[terms[0]] = terms[1]
    return result


def existsSameWord(cur_item, name):
    num=0
    for word in cur_item:
        if word in name:
            num+=1
            if num==len(cur_item)-1:
                return True
    return False


class ProperCorrector:
    def __init__(
            self,
            proper_name_path=config.proper_name_path,
            stroke_path=config.stroke_path,
            min_proper_len=2
    ):
        self.name = 'ProperCorrector'
        self.pyUtil=ChinesePinyinUtil()
        # proper name, 专名词典，包括成语、俗语、专业领域词等 format: 词语
        self.proper_names = load_set_file(self.pyUtil,proper_name_path,min_proper_len=min_proper_len)
        # stroke, 五笔笔画字典 format: 字:五笔笔画
        self.stroke_dict = load_dict_file(stroke_path)

        # tencet word2vec
        self.wss = WordSentenceSimliarity()

    def get_stroke(self, char):
        """
        取笔画
        :param char:
        :return:
        """
        return self.stroke_dict.get(char, '')

    def get_pinyin(self, char):
        return pypinyin.lazy_pinyin(char)

    def is_near_stroke_char(self, char1, char2, stroke_threshold=0.8):
        """
        判断两个字是否形似
        :param char1:
        :param char2:
        :return: bool
        """
        return self.get_char_stroke_similarity_score(char1, char2) > stroke_threshold

    def get_char_stroke_similarity_score(self, char1, char2):
        """
        获取字符的字形相似度
        Args:
            char1:
            char2:

        Returns:
            float, 字符相似度值
        """
        score = 0.0
        if char1 == char2:
            score = 1.0
        # 如果一个是中文字符，另一个不是，为0
        if is_chinese(char1) != is_chinese(char2):
            return score
        if not is_chinese(char1):
            return score
        char_stroke1 = self.get_stroke(char1)
        char_stroke2 = self.get_stroke(char2)
        # 相似度计算：1-编辑距离
        score = 1.0 - edit_distance(char_stroke1, char_stroke2)
        return score

    def get_word_stroke_similarity_score(self, word1, word2):
        """
        计算两个词的字形相似度
        :param word1:
        :param word2:
        :return: float, 相似度
        """
        if word1 == word2:
            return 1.0
        if len(word1) != len(word2):
            return 0.0
        total_score = 0.0
        for i in range(len(word1)):
            char1 = word1[i]
            char2 = word2[i]
            if not self.is_near_stroke_char(char1, char2):
                return 0.0
            char_sim_score = self.get_char_stroke_similarity_score(char1, char2)
            total_score += char_sim_score
        score = total_score / len(word1)
        return score

    def is_near_pinyin_char(self, char1, char2) -> bool:
        """
        判断两个单字的拼音是否是临近读音
        :param char1:
        :param char2:
        :return: bool
        """
        char_pinyin1 = self.get_pinyin(char1)[0]
        char_pinyin2 = self.get_pinyin(char2)[0]
        if not char_pinyin1 or not char_pinyin2:
            return False
        if len(char_pinyin1) == len(char_pinyin2):
            return True
        confuse_dict = {
            "l": "n",
            "zh": "z",
            "ch": "c",
            "sh": "s",
            "eng": "en",
            "ing": "in",
        }
        for k, v in confuse_dict.items():
            if char_pinyin1.replace(k, v) == char_pinyin2.replace(k, v):
                return True
        return False

    def get_char_pinyin_similarity_score(self, char1, char2):
        """
        获取字符的拼音相似度
        :param char1:
        :param char2:
        :return: float, 相似度
        """
        score = 0.0
        if char1 == char2:
            score = 1.0
        # 如果一个是中文字符，另一个不是，为0
        if is_chinese(char1) != is_chinese(char2):
            return score
        if not is_chinese(char1):
            return score
        char_pinyin1 = self.get_pinyin(char1)[0]
        char_pinyin2 = self.get_pinyin(char2)[0]
        # 相似度计算：1-编辑距离
        score = 1.0 - edit_distance(char_pinyin1, char_pinyin2)
        return score

    def get_word_pinyin_similarity_score(self, word1, word2):
        """
        计算两个词的拼音相似度
        :param word1:
        :param word2:
        :return: float, 相似度
        """
        if word1 == word2:
            return 1.0
        if len(word1) != len(word2):
            return 0.0
        total_score = 0.0
        for i in range(len(word1)):
            char1 = word1[i]
            char2 = word2[i]
            if not self.is_near_pinyin_char(char1, char2):
                return 0.0
            char_sim_score = self.get_char_pinyin_similarity_score(char1, char2)
            total_score += char_sim_score
        score = total_score / len(word1)
        return score

    def get_word_similarity_score(self, word1, word2):
        """
        计算两个词的相似度
        :param word1:
        :param word2:
        :return: float, 相似度
        """
        return max(
            self.get_word_stroke_similarity_score(word1, word2),
            self.get_word_pinyin_similarity_score(word1, word2)
        )

    def proper_gen(
            self,
            text,
            word_groups,
            seed=100,
            cut_type='char',
            ngram=1234,
            sim_threshold=0.90,
            max_word_length=8,
            min_word_length=2
    ):
        """
        专名纠错
        :param text: str, 待纠错的文本
        :param start_idx: int, 文本开始的索引，兼容correct方法
        :param cut_type: str, 分词类型，'char' or 'word'
        :param ngram: 遍历句子的ngram
        :param sim_threshold: 相似度得分阈值，超过该阈值才进行纠错
        :param max_word_length: int, 专名词的最大长度为4
        :param min_word_length: int, 专名词的最小长度为2
        :return: tuple(str, list), list(wrong, right, begin_idx, end_idx)
        """
        text_new = text
        detail = []
        sentence=text
        random.seed(seed)
        # 遍历句子中的所有词，专名词的最大长度为4,最小长度为2
        sentence_words = segment(sentence, cut_type=cut_type)
        ngrams = NgramUtil.ngrams(sentence_words, ngram, join_string="_")
        # 去重
        ngrams = list(set([i.replace("_", "") for i in ngrams if i]))
        # 词长度过滤
        ngrams = [i for i in ngrams if min_word_length <= len(i) <= max_word_length]
        flag=False
        skip = random.randint(0, len(ngrams) - 2)
        n=0
        for cur_item in ngrams:
            n+=1
            if n<skip:
                continue
            if cur_item not in word_groups:
                continue
            if flag:
                break
            # 获得cur_item的core_pinyin,core_pinyin_tone
            key1, key2 = getDoubleKey(self.pyUtil, cur_item)
            if key1 in self.proper_names:
                if key2 not in self.proper_names[key1]:
                    continue
                for key2_dict in self.proper_names[key1]:
                    if flag:
                        break
                    # 是否存在相同字
                    names_list = self.proper_names[key1][key2_dict]
                    for name in names_list:
                        if existsSameWord(cur_item, name) == False:
                            continue
                        # 只计算cur_item 与name 拼音相近或相同的相似度，形似忽略，以加快计算速度
                        if self.get_word_similarity_score(cur_item, name) > sim_threshold:
                            if cur_item != name:
                                cur_idx = sentence.find(cur_item)
                                text_new = sentence[:cur_idx] + name + sentence[(cur_idx + len(cur_item)):]
                                # candidates.append(new_text)
                                # flag, r_score, s_score = self.wss.doReplace(text, sentence)
                                # if s_score - r_score > 0:
                                #     continue
                                # print(sentence, "Tencent score:", name, cur_item, r_score, s_score)
                                # temp_sentence2 = text[:(idx + cur_idx + start_idx)] + name + text[(idx + cur_idx + len(cur_item) + start_idx):]
                                # print("Find sim word :", cur_item, name, self.get_word_similarity_score(cur_item, name))
                                detail=(cur_item, name, cur_idx,cur_idx + len(cur_item))
                                flag=True
                                break
            # text_new = sentence
        return text_new,detail
    def proper_correct(
            self,
            text,
            start_idx=0,
            cut_type='char',
            ngram=1234,
            sim_threshold=0.95,
            max_word_length=4,
            min_word_length=2
    ):
        """
        专名纠错
        :param text: str, 待纠错的文本
        :param start_idx: int, 文本开始的索引，兼容correct方法
        :param cut_type: str, 分词类型，'char' or 'word'
        :param ngram: 遍历句子的ngram
        :param sim_threshold: 相似度得分阈值，超过该阈值才进行纠错
        :param max_word_length: int, 专名词的最大长度为4
        :param min_word_length: int, 专名词的最小长度为2
        :return: tuple(str, list), list(wrong, right, begin_idx, end_idx)
        """
        text_new = ''
        details = []
        # 切分为短句
        sentences = split_2_short_text(text, include_symbol=True)
        for sentence, idx in sentences:
            # 遍历句子中的所有词，专名词的最大长度为4,最小长度为2
            sentence_words = segment(sentence, cut_type=cut_type)
            ngrams = NgramUtil.ngrams(sentence_words, ngram, join_string="_")
            # 去重
            ngrams = list(set([i.replace("_", "") for i in ngrams if i]))
            # 词长度过滤
            ngrams = [i for i in ngrams if min_word_length <= len(i) <= max_word_length]
            for cur_item in ngrams:
                # 获得cur_item的core_pinyin,core_pinyin_tone
                key1,key2=getDoubleKey(self.pyUtil,cur_item)
                if key1 in self.proper_names:
                    # if key2 not in self.proper_names[key1]:
                    #     continue
                    for key2_dict in self.proper_names[key1]:
                        # 是否存在相同字
                        # print(key1,key2,key2_dict)
                        names_list=self.proper_names[key1][key2_dict]
                        for name in names_list:
                            if existsSameWord(cur_item,name)==False:
                                continue
                            # 只计算cur_item 与name 拼音相近或相同的相似度，形似忽略，以加快计算速度
                            # if self.get_word_similarity_score(cur_item, name) > sim_threshold:
                            if cur_item != name:
                                cur_idx = text.find(cur_item)
                                # temp_sentence2 = text[:cur_idx] + name + text[(cur_idx + len(cur_item)):]
                                # flag,r_score,s_score=self.wss.doReplace(text, temp_sentence2)
                                # if s_score-r_score>0:
                                #     continue
                                # print(temp_sentence2,"Tencent score:",name,cur_item,r_score,s_score)
                                # temp_sentence2 = text[:(idx + cur_idx + start_idx)] + name + text[(idx + cur_idx + len(cur_item) + start_idx):]
                                print("Find sim word :",cur_item,name,self.get_word_similarity_score(cur_item, name))
                                details.append(
                                    (cur_item, name, idx + cur_idx + start_idx, idx + cur_idx + len(cur_item) + start_idx))

            text_new += sentence
        return text_new, details
