# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: corrector with pinyin and stroke
"""
import operator
import os
from codecs import open
from loguru import logger
import pypinyin

from models.mypycorrector import config
from models.mypycorrector.detector import Detector, ErrorType
from models.mypycorrector.utils.math_utils import edit_distance_word
from models.mypycorrector.utils.text_utils import is_chinese_string
from models.mypycorrector.utils.tokenizer import segment, split_2_short_text


class Corrector(Detector):
    def __init__(
            self,
            common_char_path=config.common_char_path,
            same_pinyin_path=config.same_pinyin_path,
            same_stroke_path=config.same_stroke_path,
            language_model_path=config.language_model_path,
            word_freq_path=config.word_freq_path,
            custom_word_freq_path='',
            custom_confusion_path='',
            person_name_path=config.person_name_path,
            place_name_path=config.place_name_path,
            stopwords_path=config.stopwords_path,
            proper_name_path=config.proper_name_path,
            stroke_path=config.stroke_path
    ):
        super(Corrector, self).__init__(
            language_model_path=language_model_path,
            word_freq_path=word_freq_path,
            custom_word_freq_path=custom_word_freq_path,
            custom_confusion_path=custom_confusion_path,
            person_name_path=person_name_path,
            place_name_path=place_name_path,
            stopwords_path=stopwords_path,
            proper_name_path=proper_name_path,
            stroke_path=stroke_path
        )
        self.name = 'corrector'
        self.common_char_path = common_char_path
        self.same_pinyin_text_path = same_pinyin_path
        self.same_stroke_text_path = same_stroke_path
        self.initialized_corrector = False
        self.cn_char_set = None
        self.same_pinyin = None
        self.same_stroke = None

    @staticmethod
    def load_set_file(path):
        words = set()
        with open(path, 'r', encoding='utf-8') as f:
            for w in f:
                w = w.strip()
                if w.startswith('#'):
                    continue
                if w:
                    words.add(w)
        return words

    @staticmethod
    def load_same_pinyin(path, sep='\t'):
        """
        ???????????????
        :param path:
        :param sep:
        :return:
        """
        result = dict()
        if not os.path.exists(path):
            logger.warn("file not exists:" + path)
            return result
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split(sep)
                if parts and len(parts) > 2:
                    key_char = parts[0]
                    same_pron_same_tone = set(list(parts[1]))
                    same_pron_diff_tone = set(list(parts[2]))
                    value = same_pron_same_tone.union(same_pron_diff_tone)
                    if key_char and value:
                        result[key_char] = value
        return result

    @staticmethod
    def load_same_stroke(path, sep='\t'):
        """
        ???????????????
        :param path:
        :param sep:
        :return:
        """
        result = dict()
        if not os.path.exists(path):
            logger.warn("file not exists:" + path)
            return result
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split(sep)
                if parts and len(parts) > 1:
                    for i, c in enumerate(parts):
                        exist = result.get(c, set())
                        current = set(list(parts[:i] + parts[i + 1:]))
                        result[c] = exist.union(current)
        return result

    def _initialize_corrector(self):
        # chinese common char
        self.cn_char_set = self.load_set_file(self.common_char_path)
        # same pinyin
        self.same_pinyin = self.load_same_pinyin(self.same_pinyin_text_path)
        # same stroke
        self.same_stroke = self.load_same_stroke(self.same_stroke_text_path)
        self.initialized_corrector = True

    def check_corrector_initialized(self):
        if not self.initialized_corrector:
            self._initialize_corrector()

    def get_same_pinyin(self, char):
        """
        ????????????
        :param char:
        :return:
        """
        self.check_corrector_initialized()
        return self.same_pinyin.get(char, set())

    def get_same_stroke(self, char):
        """
        ????????????
        :param char:
        :return:
        """
        self.check_corrector_initialized()
        return self.same_stroke.get(char, set())

    def known(self, words):
        """
        ???????????????????????????????????????
        :param words:
        :return:
        """
        self.check_detector_initialized()
        return set(word for word in words if word in self.word_freq)

    def _confusion_char_set(self, c):
        return self.get_same_pinyin(c).union(self.get_same_stroke(c))

    def _confusion_word_set(self, word):
        confusion_word_set = set()
        candidate_words = list(self.known(edit_distance_word(word, self.cn_char_set)))
        for candidate_word in candidate_words:
            if pypinyin.lazy_pinyin(candidate_word) == pypinyin.lazy_pinyin(word):
                # same pinyin
                confusion_word_set.add(candidate_word)
        return confusion_word_set

    def _confusion_custom_set(self, word):
        confusion_word_set = set()
        if word in self.custom_confusion:
            confusion_word_set = {self.custom_confusion[word]}
        return confusion_word_set

    def generate_items(self, word, fragment=1):
        """
        ?????????????????????
        :param word:
        :param fragment: ??????
        :return:
        """
        self.check_corrector_initialized()
        # 1???
        candidates_1 = []
        # 2???
        candidates_2 = []
        # ??????2???
        candidates_3 = []

        # same pinyin word
        candidates_1.extend(self._confusion_word_set(word))
        # custom confusion word
        candidates_1.extend(self._confusion_custom_set(word))
        # get similarity char
        if len(word) == 1:
            # sim one char
            confusion = [i for i in self._confusion_char_set(word[0]) if i]
            candidates_1.extend(confusion)
        if len(word) == 2:
            # sim first char
            confusion_first = [i for i in self._confusion_char_set(word[0]) if i]
            candidates_2.extend([i + word[1] for i in confusion_first])
            # sim last char
            confusion_last = [i for i in self._confusion_char_set(word[1]) if i]
            candidates_2.extend([word[0] + i for i in confusion_last])
            # both change, sim char
            candidates_2.extend([i + j for i in confusion_first for j in confusion_last if i + j])
            # sim word
            # candidates_2.extend([i for i in self._confusion_word_set(word) if i])
        if len(word) > 2:
            # sim mid char
            confusion = [word[0] + i + word[2:] for i in self._confusion_char_set(word[1])]
            candidates_3.extend(confusion)

            # sim first word
            confusion_word = [i + word[-1] for i in self._confusion_word_set(word[:-1])]
            candidates_3.extend(confusion_word)

            # sim last word
            confusion_word = [word[0] + i for i in self._confusion_word_set(word[1:])]
            candidates_3.extend(confusion_word)

        # add all confusion word list
        confusion_word_set = set(candidates_1 + candidates_2 + candidates_3)
        confusion_word_list = [item for item in confusion_word_set if is_chinese_string(item)]
        confusion_sorted = sorted(confusion_word_list, key=lambda k: self.word_frequency(k), reverse=True)
        return confusion_sorted[:len(confusion_word_list) // fragment + 1]

    def get_lm_correct_item(self, cur_item, candidates, before_sent, after_sent, threshold=57, cut_type='char'):
        """
        ????????????????????????????????????
        :param cur_item: ?????????
        :param candidates: ?????????
        :param before_sent: ??????????????????
        :param after_sent: ??????????????????
        :param threshold: ppl??????, ??????????????????????????????ppl?????????????????????
        :param cut_type: ????????????, ?????????
        :return: str, correct item, ???????????????
        """
        result = cur_item
        if cur_item not in candidates:
            candidates.append(cur_item)

        ppl_scores = {i: self.ppl_score(segment(before_sent + i + after_sent, cut_type=cut_type)) for i in candidates}
        sorted_ppl_scores = sorted(ppl_scores.items(), key=lambda d: d[1])

        # ????????????????????????????????????????????????
        top_items = []
        top_score = 0.0
        for i, v in enumerate(sorted_ppl_scores):
            v_word = v[0]
            v_score = v[1]
            if i == 0:
                top_score = v_score
                top_items.append(v_word)
            # ????????????????????????
            elif v_score < top_score + threshold:
                top_items.append(v_word)
            else:
                break
        if cur_item not in top_items:
            result = top_items[0]
        return result

    def correct(self, text, exclude_proper=True,exclude_low_proper=True,max_word_length=8,
                min_word_length=4,min_match_like=4,shape_score=0.85,replace_threshold=0.015,
                recall=False, check_list=None, only_proper=False, include_symbol=True, num_fragment=1, threshold=57, **kwargs):
        """
        ????????????

        ???????????????
        1. ??????????????????
        2. ????????????
        3. ????????????
        :param text: str, query ??????
        :param include_symbol: bool, ????????????????????????
        :param num_fragment: ????????????????????????, 1 / (num_fragment + 1)
        :param threshold: ??????????????????ppl??????
        :param kwargs: ...
        :return: text (str)??????????????????, list(wrong, right, begin_idx, end_idx)
        """
        text_new = ''
        details = []
        self.check_corrector_initialized()
        # ?????????????????????
        sentences = split_2_short_text(text, include_symbol=include_symbol)
        for sentence, idx in sentences:
            maybe_errors, proper_details = self.detect_sentence(sentence, idx, exclude_proper=exclude_proper,shape_score=shape_score,
                                                                replace_threshold=replace_threshold,
                                                                exclude_low_proper=exclude_low_proper,max_word_length=max_word_length,
                                                                min_word_length=min_word_length,min_match_like=min_match_like,
                                                                recall=recall, only_proper=only_proper, check_list=check_list, **kwargs)
            for cur_item, begin_idx, end_idx, err_type in maybe_errors:
                # ?????????????????????
                before_sent = sentence[:(begin_idx - idx)]
                after_sent = sentence[(end_idx - idx):]

                # ??????????????????????????????????????????
                if err_type == ErrorType.confusion:
                    corrected_item = self.custom_confusion[cur_item]
                elif err_type == ErrorType.proper:
                    # ???????????? proper_details format: (error_word, corrected_word, begin_idx, end_idx)
                    corrected_item = [i[1] for i in proper_details if cur_item == i[0] and begin_idx == i[2]][0]
                else:
                    # ??????????????????????????????????????????
                    candidates = self.generate_items(cur_item, fragment=num_fragment)
                    if not candidates:
                        continue
                    corrected_item = self.get_lm_correct_item(
                        cur_item,
                        candidates,
                        before_sent,
                        after_sent,
                        threshold=threshold
                    )
                # output
                if corrected_item != cur_item:
                    sentence = before_sent + corrected_item + after_sent
                    detail_word = (cur_item, corrected_item, begin_idx, end_idx)
                    details.append(detail_word)
            text_new += sentence
        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details
