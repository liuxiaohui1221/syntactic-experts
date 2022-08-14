# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 专名纠错，包括成语纠错、人名纠错、机构名纠错、领域词纠错等
"""
import itertools
import json
import os
import random
from codecs import open
from fnmatch import fnmatch

import gensim
import pypinyin
from loguru import logger
from tqdm import tqdm

from models.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf
from models.mypycorrector import config
from knowledgebase.chinese_pinyin_util import ChinesePinyinUtil
from knowledgebase.tencent.SentenceSimilarity import WordSentenceSimliarity
from models.mypycorrector.utils.math_utils import edit_distance
from models.mypycorrector.utils.ngram_util import NgramUtil
from models.mypycorrector.utils.string_util import getTwoTextEdits
from models.mypycorrector.utils.text_utils import is_chinese
from models.mypycorrector.utils.tokenizer import segment, split_2_short_text
from collections import defaultdict

def getNoTonePyin(pyUtil,p_tone):
    if pyUtil.isNumber(p_tone[-1]):
        return p_tone[:-1]
    return p_tone


def getMatchDoubleKeys(pyUtil, word_group,min_proper_match_len=4,max_proper_match_len=8):
    iter = 1
    if len(word_group) >= min_proper_match_len and len(word_group)<=max_proper_match_len:
        iter = len(word_group)+1
    pysTone = pypinyin.lazy_pinyin(word_group, errors='ignore', style=pypinyin.Style.TONE3)
    core_no_tones, core_tones = [], []
    for i in range(iter):
        corePysNoTone = []
        corePysTone = []
        for j,p_tone in enumerate(pysTone):
            p_no_tone = getNoTonePyin(pyUtil, p_tone)
            cpyinNoTone = pyUtil.handleSimPinyinToCore(p_no_tone)
            cpyinTone = pyUtil.handleSimPinyinToCore(p_tone)
            if iter>1:
                if i==j:
                    corePysNoTone.append('*')
                    corePysTone.append('*')
                else:
                    corePysNoTone.append(cpyinNoTone)
                    corePysTone.append(cpyinTone)
            else:
                corePysNoTone.append(cpyinNoTone)
                corePysTone.append(cpyinTone)
        key_core_no_tone = "_".join(corePysNoTone)
        key_core_tone = "_".join(corePysTone)
        core_no_tones.append(key_core_no_tone)
        core_tones.append(key_core_tone)
    return core_no_tones, core_tones


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

# 解决多音字问题
def getMultiTonePyinKey(pyUtil,word_group):
    arr = []
    for w in word_group:
        a1 = pypinyin.pinyin(w, errors='ignore', heteronym=True, style=pypinyin.TONE3)
        arr.extend(a1)
    pysTone = list(itertools.product(*arr))
    key_notones=[]
    key_tones=[]
    for p_tones in pysTone:
        corePysNoTone = []
        corePysTone = []
        for p_tone in p_tones:
            p_no_tone = getNoTonePyin(pyUtil, p_tone)
            cpyinNoTone = pyUtil.handleSimPinyinToCore(p_no_tone)
            cpyinTone = pyUtil.handleSimPinyinToCore(p_tone)
            corePysNoTone.append(cpyinNoTone)
            corePysTone.append(cpyinTone)
        key_core_no_tone = "_".join(corePysNoTone)
        key_core_tone = "_".join(corePysTone)
        key_notones.append(key_core_no_tone)
        key_tones.append(key_core_tone)
    return set(key_notones),key_tones

def getMappingProper(pyUtil,words,min_proper_match_len=2,max_proper_match_len=8):
    # 格式：{core_pin_yin:{ core_pin_yin_tone:[词组] }}
    # 四字短语存四份，以加快模糊检索
    corePyinDB=defaultdict(dict)
    for word_group in tqdm(words):
        if len(word_group)<2:
            continue
        key_core_no_tones,key_core_tones=getMatchDoubleKeys(pyUtil,word_group,min_proper_match_len=min_proper_match_len,max_proper_match_len=max_proper_match_len)
        for index,key_core_no_tone in enumerate(key_core_no_tones):
            key_core_tone=key_core_tones[index]
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


def load_set_file(pyUtil,path,min_proper_match_len=2,max_proper_match_len=8):
    print("Loading proper file:","os.path.dirname(os.path.realpath(__file__))=%s" % os.path.dirname(path))
    proper_path=os.path.join(os.path.dirname(os.path.realpath(path)),"json_"+os.path.basename(path))
    if os.path.exists(proper_path):
        corePyins=json.load(open(proper_path,encoding='utf-8'))
        return corePyins
    else:
        words = []
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for w in f:
                    w = w.strip()
                    if w.startswith('#'):
                        continue
                    if w:
                        words.append(w)
        # 转换并保存
        corePyins=getMappingProper(pyUtil,words,min_proper_match_len=min_proper_match_len,max_proper_match_len=max_proper_match_len)
        json.dump(corePyins,open(proper_path, 'w', encoding='utf-8'),ensure_ascii=False, indent=4)
        print("Saved proper over!",proper_path)
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


def existsSameWord(cur_item, name, tolerate_count=1):
    num=0
    if len(cur_item)==len(name):
        if len(cur_item)==3:
            equal_num=len(name)-1
        else:
            equal_num=len(cur_item)-tolerate_count
        for index,word in enumerate(cur_item):
            if word in name[index]:
                num+=1
                if num==equal_num:
                    return True
    else:
        # 匹配缺字多字问题
        if len(cur_item)<len(name):
            # 缺字要求全部相等
            for match_word in cur_item:
                if match_word not in name:
                    return False
            return True
    return False


def load_stop_check_file(path):
    result = []
    if path:
        if not os.path.exists(path):
            logger.warning('file not found.%s' % path)
            return result
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    result.append(line)
    unique_words=set(result)
    print("Loaded proper stopwords:",len(unique_words))
    return unique_words


def load_chengyu_file(chengyu_path):
    return load_stop_check_file(chengyu_path)


def load_low_chengyu_file(path):
    low_words_confusion=defaultdict()
    if path:
        if not os.path.exists(path):
            logger.warning('file not found.%s' % path)
            return low_words_confusion
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line)<=1:
                        continue
                    temparr=line.split(sep='\t')
                    low_words_confusion[temparr[0]]=temparr[1]
    print("Loaded low proper:",len(low_words_confusion))
    return low_words_confusion


class ProperCorrector:
    def __init__(
            self,
            proper_name_path=config.proper_name_path,
            stroke_path=config.stroke_path,
            proper_stopwords=config.proper_stopwords,
            chengyu_path=config.chengyu_path,
            low_chengyu_path=config.low_chengyu_path,
            stopwords=None
    ):
        self.name = 'ProperCorrector'
        self.pyUtil=ChinesePinyinUtil()
        # proper name, 专名词典，包括成语、俗语、专业领域词等 format: 词语
        self.proper_names = load_set_file(self.pyUtil,proper_name_path)
        # stroke, 五笔笔画字典 format: 字:五笔笔画
        self.stroke_dict = load_dict_file(stroke_path)
        # tencet word2vec
        self.wss = WordSentenceSimliarity()
        self.stop_check_words = load_stop_check_file(proper_stopwords)
        # 区分度高的成语库
        self.chengyu_dict = load_chengyu_file(chengyu_path)
        # 区分度低的成语库
        self.low_chengyu_dict = load_low_chengyu_file(low_chengyu_path)
        self.stopwords=stopwords

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
            cut_type='char',
            ngram=1234,
            sim_threshold=0.85,
            max_word_length=4,
            min_word_length=2,
            seed=999
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
        random.seed(seed)
        text_new = text
        details = []
        sentence=text
        # 遍历句子中的所有词，专名词的最大长度为4,最小长度为2
        sentence_words = segment(sentence, cut_type=cut_type)
        ngrams = NgramUtil.ngrams(sentence_words, ngram, join_string="_")
        # 去重
        ngrams = list(set([i.replace("_", "") for i in ngrams if i]))
        # 词长度过滤
        ngrams = [i for i in ngrams if min_word_length <= len(i) <= max_word_length]
        flag=False
        n=0
        for cur_item in ngrams:
            n+=1
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
                                details.append((cur_item, name, cur_idx,cur_idx + len(cur_item)))
                                if len(details)>20:
                                    flag=True
                                    break
            # text_new = sentence
        if len(details)>0:
            index=random.randint(0,len(details)-1)
            idx_tuple=details[index]
            name=idx_tuple[1]
            cur_idx_start=idx_tuple[2]
            cur_idx_end=idx_tuple[3]
            text_new = sentence[:cur_idx_start] + name + sentence[cur_idx_end:]
            return text_new,details[index]
        return text_new,details

    def excludeProper(self, cur_item, param):
        # 排除cur_item本身为专名词:均存在则为专名词
        flag = False
        for names_list in param.values():
            if cur_item in names_list:
                flag = True
                break
        if self.excludeCheckTencentWords(cur_item)==False and self.wss.existTencentWord(cur_item):
            flag=True
        return flag
    def proper_correct(
            self,
            text,
            start_idx=0,
            cut_type='word',
            ngram=84,
            sim_threshold=0.95,
            max_word_length=8,
            min_word_length=4,
            max_match_count=1,
            check_list=None
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
        if len(text)<=1:
            return text,[]
        text_new = ''
        details = []
        correct_edits = None
        correct_words = None
        # 切分为短句
        sentences = split_2_short_text(text, include_symbol=True)
        for sentence, idx in sentences:
            if check_list and len(check_list)>0:
                flag=False
                for candidate_word in check_list:
                    if candidate_word in sentence:
                        flag=True
                        break
                if flag==False:
                    # print("Skip sentence:",sentence," not candidate:",check_list)
                    continue
            # 遍历句子中的所有词，专名词的最大长度为4,最小长度为2
            sentence_words = segment(sentence, cut_type=cut_type)
            ngrams = NgramUtil.ngrams(sentence_words, ngram, join_string="_")
            ngrams_list=[i.replace("_", "") for i in ngrams if i]
            # 去重
            ngrams_set = list(set(ngrams_list))
            # 保持原有循序
            ngrams_set.sort(key=ngrams_list.index)
            # 词长度过滤
            ngrams = [i for i in ngrams_set if min_word_length <= len(i) <= max_word_length]
            stop=False
            match_count = 0
            for cur_item in ngrams:
                if self.existStopWord(cur_item):
                    # print("ignore contains stop word:",cur_item)
                    continue
                # 过滤对应低频专名词中的混淆词
                if cur_item in self.low_chengyu_dict:
                    continue
                multi_key1, multi_key2 = getMultiTonePyinKey(self.pyUtil, cur_item)
                # key1, _key2 = getDoubleKey(self.pyUtil, cur_item)
                # 排除本身为proper的
                candidate_names=[]
                stop_check=False
                for key1 in multi_key1:
                    key1_proper=self.proper_names.get(key1,None)
                    if key1_proper:
                        flag = self.excludeProper(cur_item,key1_proper)
                        if flag:
                            stop_check=True
                            break
                        else:
                            #4字及以上容纳两个音近错误
                            for names in key1_proper.values():
                                for name in names:
                                    if existsSameWord(cur_item, name, 1) == False:
                                        continue
                                    candidate_names.append(name)
                if stop_check:
                    continue
                # 4字及以上容纳一个字错误或缺失
                candidate_one_word_names=self.findConfusionNames(cur_item,self.proper_names,multi_key1)
                isSimPyin=False
                if len(candidate_names)>0:
                    isSimPyin = True
                candidate_names.extend(candidate_one_word_names)
                for name in candidate_names:
                    sim_score=self.get_word_similarity_score(cur_item, name)
                    # 继续判断是否为成语
                    isChengyu=self.checkChengYu(name)
                    if isChengyu:
                        sim_score=2
                        print("chengyu replace:",cur_item,name)
                    if sim_score > sim_threshold:
                        if sim_score<1.1:
                            # 再次检查，排除区分度低的成语：属于区分度低的成语集中
                            lowrecg_flag=self.isLowRecgName(isSimPyin,name)
                            if lowrecg_flag:
                                # 由于候选成语优先音近检查，当前不合要求，后续跳过
                                break
                        if cur_item != name:
                            match_count+=1
                            cur_idx = sentence.find(cur_item)
                            if cur_idx==-1:
                                # 替换前面已经修改的字，再次查找
                                edits_idx_start=cur_item.find(correct_edits[0][1])
                                cur_item = cur_item[:edits_idx_start] + correct_edits[0][2] + cur_item[(edits_idx_start+len(correct_edits[0][1])):]
                                cur_idx = sentence.find(cur_item)
                            else:
                                correct_edits = getTwoTextEdits(cur_item, name)
                            # print("Find replace:", sim_score,cur_item,name)
                            sentence = sentence[:cur_idx] + name + sentence[(cur_idx + len(cur_item)):]
                            details.append(
                                (cur_item, name, idx + cur_idx + start_idx, idx + cur_idx + len(cur_item) + start_idx))
                            if match_count >= max_match_count:
                                break
                    else:
                        # print("Filter low score:",sim_score,cur_item,name,sentence)
                        pass
            text_new += sentence
        return text_new, details

    def findConfusionNames(self, cur_item, proper_names, multi_key1):
        if len(cur_item)<3:
            return []
        if self.existStopWord(cur_item):
            return None
        query_keys=self.getQueryKey(multi_key1,len(cur_item))
        return self.findCandidateNamesBySimPyinQuery(cur_item,query_keys,proper_names)

    # key1格式示例：yi_zhi_du_xiu, key2_tone格式:yi1_zhi1_du2_xiu4
    def getQueryKey(self, multi_key1, word_len):
        match_pyins = []
        for key1 in multi_key1:
            pyins=key1.split(sep='_')
            if word_len>3:
                for i in range(len(pyins),-1,-1):
                    # 4字及以上一个字缺失
                    match_pyins.append(pyins[:i] + ['*'] + pyins[i:])
                for i in range(len(pyins)-1,-1,-1):
                    # 4字及以上一个字错误
                    match_pyins.append(pyins[:i] + ['*'] + pyins[i + 1:])
        query_strs=[]
        for index,m_pys in enumerate(match_pyins):
            query_strs.append("_".join(m_pys))
        return query_strs

    def isNumber(self, c):
        return c >= '0' and c <= '9'

    def findCandidateNamesBySimPyinQuery(self, cur_item, query_keys, proper_names):
        candidate_names=[]
        for query in query_keys:
                for names in proper_names.get(query,{}).values():
                    for name in names:
                        if cur_item==name or len(name)<=1:
                            continue
                        # 一个字缺失或错误的，则需满足其余字相等
                        if existsSameWord(cur_item, name) == False:
                            continue
                        # 使用tencent词向量再次过滤非专名词
                        if self.wss.existTencentWord(name)==False:
                            continue
                        candidate_names.append(name)
        return set(candidate_names)

    def existStopWord(self, cur_item):
        flag=False
        for w in self.stop_check_words:
            if w in cur_item:
                flag=True
                break
        if flag==False:
            for c_w in cur_item:
                if is_chinese(c_w)==False:
                    flag=True
                    break

        # if flag==False and self.stopwords:
        #     for stop_w in self.stopwords:
        #         if stop_w in cur_item:
        #             flag=True
        #             break
        return flag

    def existsZiMu(self, w):
        if (w <= 'Z' and w >= 'A') or (w <= 'z' and w >= 'a') :
            return True
        return False

    def fastMatch(self, query, py_key1):
        for index,pyin in enumerate(query):
            if pyin=='*':
                continue
            if pyin!=py_key1[index]:
                return False
        return True

    def excludeCheckTencentWords(self, cur_item):
        words=['人民检察院']
        for word in words:
            if cur_item in word:
                return True
        return False

    def checkChengYu(self, name):
        if name in self.chengyu_dict:
            return True
        return False

    def isLowRecgName(self, isSimPyin,name):
        if isSimPyin==False and name in self.low_chengyu_dict.values():
            return True
        return False