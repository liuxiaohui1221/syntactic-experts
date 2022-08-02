# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description:
"""
import argparse
import operator
import sys
import time
import os
from difflib import SequenceMatcher
from operator import itemgetter

import numpy
from transformers import BertTokenizer, BertForMaskedLM
import torch
from typing import List
from loguru import logger

from data_augmentation.preliminary_gen import isChinese
from knowledgebase.chinese_pinyin_util import ChinesePinyinUtil
from knowledgebase.chinese_shape_util import ChineseShapeUtil
from knowledgebase.tencent.SentenceSimilarity import WordSentenceSimliarity
from model.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf

sys.path.append('../..')
from pycorrector import config
from pycorrector.utils.tokenizer import split_text_by_maxlen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤', '\t', '֍', '玕', '', '↓','◆']


def findPos(corrected_text, start,words):
    position = corrected_text.find(words)
    num=0
    while position != -1:
        if position>start:
            break
        num+=1
        position = corrected_text.find(words, position + 1)
    return num

class MacBertCorrector(object):
    def __init__(self, model_dir=config.macbert_model_dir):
        self.name = 'macbert_corrector'
        t1 = time.time()
        bin_path = os.path.join(model_dir, 'pytorch_model.bin')
        if not os.path.exists(bin_path):
            model_dir = "shibing624/macbert4csc-base-chinese"
            logger.warning(f'local model {bin_path} not exists, use default HF model {model_dir}')

        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForMaskedLM.from_pretrained(model_dir)
        self.model.to(device)
        self.thu=VocabConf().thulac_singleton
        self.pyUtil=ChinesePinyinUtil()
        self.shapeUtil=ChineseShapeUtil()
        self.word2vecSim=WordSentenceSimliarity(self.thu)
        logger.debug("Use device: {}".format(device))
        logger.debug('Loaded macbert4csc model: %s, spend: %.3f s.' % (model_dir, time.time() - t1))

    def macbert_correct(self, text,val_target=None):
        """
        句子纠错
        :param text: 句子文本
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        text_new = ''
        details = []
        # 长句切分为短句
        blocks = split_text_by_maxlen(text, maxlen=128)
        block_texts = [block[0] for block in blocks]
        inputs = self.tokenizer(block_texts, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        for ids, (text, idx) in zip(outputs.logits, blocks):
            decode_tokens = self.tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
            corrected_text = decode_tokens[:len(text)]
            corrected_text, sub_details = self.get_errors(corrected_text, text)
            text_new += corrected_text
            sub_details = [(i[0], i[1], idx + i[2], idx + i[3]) for i in sub_details]
            details.extend(sub_details)
        self.macbert_correct_recall(text,val_target=val_target, first_predict=text_new)
        return text_new, details
    # 检错纠错召回
    def macbert_correct_recall(self, text, val_target=None, first_predict=None, topk=10):
        """
        句子纠错
        :param text: 句子文本
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        text_word_recalls = []
        details = []
        # 长句切分为短句
        blocks = split_text_by_maxlen(text, maxlen=128)
        block_texts = [block[0] for block in blocks]
        if len(block_texts)>1:
            print("splits: ",block_texts)
        inputs = self.tokenizer(block_texts, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        for ids, (text, idx) in zip(outputs.logits, blocks):
            candidate_correcteds=self.getTopKIds(ids.clone(),topk)
            np_candidate_corrected=numpy.array(candidate_correcteds)
            # print(candidate_correcteds)
            np_candidate_corrected=np_candidate_corrected.transpose()
            # print("转置：",np_candidate_corrected)
            #
            # for pos,word_candidates_ids in enumerate(np_candidate_corrected):# id转token
            #     word_candidates=word_candidates_ids
            #     # word_candidates=self.tokenizer.decode(word_candidates_ids, skip_special_tokens=False).replace(' ', '')
            #     text_word_recalls.append(word_candidates)
        # 删除special_token所在行
        text_word_recalls=self.deleteSpecialTokens(np_candidate_corrected,topk)
        # print("".join(text_word_recalls[:,0]))
        # 1、召回候选集：模型topK,同音近形集，混淆集 （只考虑模型的纠错字位置？）
        src_err_words,topkRecallPerErrPos=self.findErrorWrods(text,text_word_recalls,topk)
        # 2.召回粗选：a.模型topK中：选择top1，top2-K中选同音近形的；b.同音近形中选与原句上下字组成词语的; c.原句中错字存在对应混淆集的
        err_word_pinyin_candidates,err_word_recalltopk_candidates=self.findCandidateWordsPerErrPos(
            text,src_err_words,topkRecallPerErrPos)
        # 计算跨语义替换得分
        isSem=self.isSemanticReplace(err_word_pinyin_candidates,err_word_recalltopk_candidates)
        if isSem:
            # 若topK召回集中有近音形的则优先从里面选
            # 3.计算替换得分
            scores=self.computeReplaceScore(text,err_word_pinyin_candidates,err_word_recalltopk_candidates)
            # 返回格式：[(srcWord,pos,replaceWord),,,,]
            choosed_replace_words=self.chooseBestCandidate(err_word_pinyin_candidates,err_word_recalltopk_candidates,scores)
            print("Scores:",scores)
        if val_target and err_word_pinyin_candidates:
            actual_edits=self.getTwoTextEdits(text,val_target)
            print("Actual edits: ",actual_edits," Found candidates(topK,pinyin_shape): ",err_word_recalltopk_candidates,err_word_pinyin_candidates)
        return text_word_recalls, details

    def findErrorWrods(self,text, text_word_recalls, topk):
        correct_top1_ids = text_word_recalls[:, 0]
        decode_tokens=self.tokenizer.decode(correct_top1_ids,skip_special_tokens=True).replace(' ', '')
        corrected_text = decode_tokens[:len(text)]
        correct_top1, sub_details = self.get_errors(corrected_text, text)
        r = SequenceMatcher(None, text, correct_top1)
        diffs = r.get_opcodes()
        text_edits = []
        candidates = numpy.empty(shape=[0, topk], dtype=numpy.str_)
        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if tag != "replace" and tag!="equal":
                return None,None
            if tag=="equal":
                continue
            # 过滤非汉字
            flag = False
            for w in text[i1:i2]:
                if isChinese(w) == False:
                    flag = True
                    break
            if flag:
                continue
            text_edits.append((tag, text[i1:i2], i1, i2))

            word_ids=self.tokenizer.convert_tokens_to_ids(list(correct_top1[j1:j2]))
            slice_start,slice_end=self.findSliceWordCandidates(corrected_text,i1,corrected_text[i1:i2],correct_top1_ids,word_ids)
            candidates = numpy.append(candidates, text_word_recalls[slice_start:slice_end], axis=0)
        return text_edits, candidates

    def isSemanticReplace(self,err_word_pinyin_candidates, err_word_recalltopk_candidates):
        if err_word_recalltopk_candidates==None:
            return False
        for words in err_word_recalltopk_candidates:
            r_word=words[0]
            flag = False
            for e_word_candidates in err_word_pinyin_candidates:
                if r_word in e_word_candidates[2]:
                    flag = True
            if flag == False:
                return True
        return False
    def batch_macbert_correct(self, texts: List[str], max_length: int = 128):
        """
        句子纠错
        :param texts: list[str], sentence list
        :param max_length: int, max length of each sentence
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        result = []

        inputs = self.tokenizer(texts, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        for ids, text in zip(outputs.logits, texts):
            text_new = ''
            details = []
            decode_tokens = self.tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
            corrected_text = decode_tokens[:len(text)]
            corrected_text, sub_details = self.get_errors(corrected_text, text)
            text_new += corrected_text
            sub_details = [(i[0], i[1], i[2], i[3]) for i in sub_details]
            details.extend(sub_details)
            details.extend(sub_details)
            result.append([text_new, details])
        return result

    def get_errors(self,corrected_text, origin_text):
        sub_details = []
        for i, ori_char in enumerate(origin_text):
            if i >= len(corrected_text):
                continue
            if ori_char in unk_tokens:
                # deal with unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            # # 忽略英文字母的大小写纠错
            # if ori_char.lower() == corrected_text[i].lower():
            #     corrected_text = corrected_text[:i] + ori_char + corrected_text[i+1:]
            #     continue
            if ori_char != corrected_text[i]:
                if ori_char.lower() == corrected_text[i]:
                    # pass english upper char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    def deleteSpecialTokens(self, text_word_recalls, topk):
        fine_recalls = numpy.empty(shape=[0, topk], dtype=numpy.str_)
        for index, word_candidates in enumerate(text_word_recalls[:]):
            if word_candidates[0] in self.tokenizer.all_special_ids:
                continue
            fine_recalls = numpy.append(fine_recalls, [word_candidates], axis=0)
        return fine_recalls

    def getTwoTextEdits(self,src_text, m1_text):
        r = SequenceMatcher(None, src_text, m1_text)
        diffs = r.get_opcodes()
        m1_edits = []
        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if "equal" in tag:
                continue
            m1_edits.append((tag, src_text[i1:i2], m1_text[j1:j2]))
        return m1_edits
    def getTopKIds(self, ids, topk):
        topCandidates=[]
        for i in range(topk):
            # 每次取top1 id，并重置为最小值
            takeTopText=torch.argmax(ids, dim=-1)

            for index,mid in enumerate(takeTopText.cpu().numpy()):
                ids[index][mid]=-1000000
            topCandidates.append(takeTopText.cpu().numpy().tolist())
        return topCandidates

    def findCandidateWordsPerErrPos(self, src_text, src_err_words, topkRecallPerErrPos,thresh=0.7):
        if src_err_words==None or len(src_err_words)==0:
            return None,None
        candidate_confusions = self.getConfusionsPerErrWords(src_err_words)
        candidate_pin_shapes = self.getPinShapePerErrWords(src_text,src_err_words,thresh=thresh)
        candidate_pin_shapes_from_topk = self.filterByPinShape(src_text,src_err_words,topkRecallPerErrPos,thresh=thresh)
        # print(src_err_words,candidate_pin_shapes,candidate_pin_shapes_from_topk)

        return candidate_pin_shapes,candidate_pin_shapes_from_topk
    def getConfusionsPerErrWords(self, src_err_words):
        # todo
        return None

    def getPinShapePerErrWords(self, src_text, src_err_words,matchGroup=False,thresh=0.7):
        error_word_candidates=[]
        for wordTuple in src_err_words:
            for index,wordErr in enumerate(wordTuple[1]):
                simChineses=self.pyUtil.getSimilarityChineseBySimPinyin(wordErr)
                simChineses.extend(self.shapeUtil.getAllSimilarityShape(wordErr,thresh=thresh))
                if matchGroup:
                    # 逐一判断是否在原句组成词组
                    candidateSimWords = []
                    for simWord in simChineses:
                        replace_text=src_text[:wordTuple[2]+index]+simWord+src_text[wordTuple[2]+index+1:]
                        splits_words=self.thu.cut(replace_text)
                        size=0

                        for s_words in splits_words:
                            size+=len(s_words[0])
                            if size<=wordTuple[2]+index:
                                continue
                            if simWord in s_words[0]  and len(s_words[0])>1:
                                candidateSimWords.append(simWord)
                            break
                    error_word_candidates.append((wordTuple[1][index],wordTuple[2]+index,candidateSimWords))
                else:
                    error_word_candidates.append((wordTuple[1][index],wordTuple[2]+index,simChineses))
        return error_word_candidates
    def filterByPinShape(self,src_text, src_err_words, topkRecallPerErrPos,thresh=0.7):
        simChineses=[]
        for wordTuple in src_err_words:
            for index, wordErr in enumerate(wordTuple[1]):
                simPinChineses = self.pyUtil.getSimilarityChineseBySimPinyin(wordErr)
                simShapeChineses= self.shapeUtil.getAllSimilarityShape(wordErr,thresh=thresh)
                simChineses.append(simPinChineses)
                if len(simShapeChineses)>0:
                    simChineses.append(simShapeChineses)
        topkFineRecall=[]
        for err_i in range(wordTuple[3]-wordTuple[2]):
            # print(src_text,src_err_words,topkRecallPerErrPos)
            correct_topk_ids = topkRecallPerErrPos[err_i,:]
            decode_tokens = self.tokenizer.decode(correct_topk_ids, skip_special_tokens=True).replace(' ', '')
            correct_topk, sub_details = self.get_errors(decode_tokens, src_text)
            topkCandidates=correct_topk[1:]
            simCandidates=set(simChineses[err_i])
            topi_candidates=[correct_topk[0]]
            for t in topkCandidates:
                for s in simCandidates:
                    if t==s:
                        topi_candidates.extend(t)
                        break
            topkFineRecall.append(topi_candidates)
        return topkFineRecall

    def findSliceWordCandidates(self, corrected_text, start,words, correct_top1_ids, word_ids):
        num=findPos(corrected_text,start,words)
        count,pos_s=0,0
        word_ids_len=0
        for i,ids in enumerate(correct_top1_ids):
            pos_s=i
            if isinstance(word_ids,int):
                word_ids_len=1
                if int(ids) == word_ids:
                    count+=1
            else:
                word_ids_len=len(word_ids)
                if int(ids) != word_ids[0]:
                    continue
                flag = True
                for j in range(len(word_ids)):
                    if int(correct_top1_ids[pos_s+j])!=word_ids[j]:
                        flag=False
                        break
                if flag:
                    count+=1
            if count == num:
                break
        return pos_s,pos_s+word_ids_len

    def computeReplaceScore(self, text, err_word_pinyin_candidates, err_word_recalltopk_candidates):
        if err_word_pinyin_candidates==None:
            return None
        word_scores={}
        for tuple3 in err_word_pinyin_candidates:
            pos=tuple3[1]
            candidate_words=tuple3[2]
            r_word_scores={}
            for r_word in candidate_words:
                new_text=text[:pos]+r_word+text[pos+1:]
                flag,r_score,s_score=self.word2vecSim.doReplace(text,new_text)
                r_word_scores[r_word]=r_score
            word_scores[tuple3[0]]=r_word_scores
        for index,word_candidates in enumerate(err_word_recalltopk_candidates):
            pos=err_word_pinyin_candidates[index][1]
            for r_word in word_candidates:
                new_text = text[:pos] + r_word + text[pos + 1:]
                flag, r_score, s_score = self.word2vecSim.doReplace(text, new_text)
                r_word_scores[r_word] = r_score

        sorted_scores={}
        for key,values in word_scores.items():
            sorted_similarity = sorted(values.items(), key=itemgetter(1), reverse=True)
            sorted_scores[key]=sorted_similarity
        return sorted_scores

    def chooseBestCandidate(self, err_word_pinyin_candidates, err_word_recalltopk_candidates, scores):
        for topk_candidates in err_word_pinyin_candidates:
            if len(topk_candidates)<=1:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--macbert_model_dir", default='output/macbert4csc',
                        type=str,
                        help="MacBert pre-trained model dir")
    args = parser.parse_args()

    m = MacBertCorrector(args.macbert_model_dir)
    error_sentences = [
        '#文明礼仪微讲堂#（四）四、公务礼仪一当面接待扎仪上级来访，接待要周到。'
    ]
    target_sentences=[
        '#文明礼仪微讲堂#（四）四、公务礼仪一当面接待礼仪上级来访，接待要周到。'
    ]
    t1 = time.time()
    for sent in error_sentences:
        corrected_sent, err = m.macbert_correct(sent,val_target=target_sentences[0])
        print("original sentence:{} => {} err:{}".format(sent, corrected_sent, err))
    print('[single]spend time:', time.time() - t1)
    t2 = time.time()
    res = m.batch_macbert_correct(error_sentences)
    for sent, r in zip(error_sentences, res):
        print("original sentence:{} => {} err:{}".format(sent, r[0], r[1]))
    print('[batch]spend time:', time.time() - t2)
