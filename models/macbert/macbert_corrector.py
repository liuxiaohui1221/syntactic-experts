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
from pypinyin import Style, pinyin, lazy_pinyin
from transformers import BertTokenizer, BertForMaskedLM
import torch
from typing import List
from loguru import logger
from collections import Counter
from data_augmentation.preliminary_gen import isChinese
from knowledgebase.chinese_pinyin_util import ChinesePinyinUtil
from knowledgebase.chinese_shape_util import ChineseShapeUtil
from knowledgebase.tencent.SentenceSimilarity import WordSentenceSimliarity
from models.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf

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
        self.thu=VocabConf().jieba_singleton
        self.pyUtil=ChinesePinyinUtil()
        self.shapeUtil=ChineseShapeUtil()
        self.word2vecSim=WordSentenceSimliarity(self.thu)
        logger.debug("Use device: {}".format(device))
        logger.debug('Loaded macbert4csc model: %s, spend: %.3f s.' % (model_dir, time.time() - t1))

    def macbert_correct(self, text,val_target=None):

        # 优先按逗号分隔
        # texts=self.split_by_douhao(text)
        # print("splits:",texts)
        """
        句子纠错
        :param text: 句子文本
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        text_new = ''
        details = []
        # 长句切分为短句
        blocks = split_text_by_maxlen(text, maxlen=256)
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
        # text_new,details2=self.macbert_correct_recall(text,text_new,val_target=val_target, first_predict=text_new)
        return text_new, details
    # 检错纠错召回
    def macbert_correct_recall(self, text,val_target=None, first_predict=None, topk=30):
        """
        句子纠错
        :param text: 句子文本
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        # 优先按逗号分隔
        # texts = self.split_by_douhao(stext)
        # print("split:",texts)
        text_word_recalls = []
        details = []
        # 长句切分为短句
        blocks = split_text_by_maxlen(text, maxlen=128)
        block_texts = [block[0] for block in blocks]
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
        correct_top1_ids = text_word_recalls[:, 0]
        # 统计topK召回率
        isRecalled=None
        pos=None
        if val_target:
            if len(correct_top1_ids)==len(val_target):
                isRecalled,pos=self.computeTopk(correct_top1_ids,text_word_recalls[:, 0:topk],val_target)
                print("recall: ",isRecalled,pos)
            else:
                print("different length:",len(correct_top1_ids),len(val_target))
        decode_tokens = self.tokenizer.decode(correct_top1_ids, skip_special_tokens=True).replace(' ', '')
        corrected_text_first_decode = decode_tokens[:len(text)]
        corrected_text_first, details = self.get_errors(corrected_text_first_decode, text)
        #
        # 1、召回候选集：模型topK,同音近形集，混淆集 （只考虑模型的纠错字位置？）
        # src_err_words格式：[(tag,待替换字,i1,i2,替换字,j1,j2),(...),,,]
        src_err_words,topkRecallPerErrPos=self.findErrorWrods(text,text_word_recalls,corrected_text_first_decode,topk)
        if len(src_err_words)==0:
            return corrected_text_first,None,[isRecalled,pos]
        # 2.召回粗选：a.模型topK中：选择top1，top2-K中选同音近形的；b.同音近形中选与原句上下字组成词语的; c.原句中错字存在对应混淆集的
        err_word_pinyin_candidates,err_word_recalltopk_candidates=self.findCandidateWordsPerErrPos(
            text,src_err_words,topkRecallPerErrPos)

        # 判断是否为跨语义预测，若是跨语义则考虑同音近音近形召回排序
        isSem=self.isSemanticReplace(err_word_pinyin_candidates,err_word_recalltopk_candidates,src_err_words)
        if isSem:
            # 若topK召回集中有近音形的则优先从里面选
            # 3.计算替换得分
            scores=self.computeReplaceScore(text,err_word_pinyin_candidates,err_word_recalltopk_candidates)
            # 返回格式：[(srcWord,pos,replaceWord),,,,]
            choosed_text=self.chooseBestCandidate(text,err_word_pinyin_candidates,scores)
            corrected_text, details = self.get_errors(choosed_text[0], text)
            # print("Choosed:",choosed_text,firstCandidates,scores)
            return corrected_text, choosed_text[1],[isRecalled,pos]
        else:
            corrected_text, details = self.get_errors(corrected_text_first, text)
            return corrected_text,details,[isRecalled,pos]
    def findErrorWrods(self,text, text_word_recalls, corrected_text, topk):
        correct_top1_ids = text_word_recalls[:, 0]
        r = SequenceMatcher(None, text, corrected_text)
        diffs = r.get_opcodes()
        text_edits = []
        candidates = numpy.empty(shape=[0, topk], dtype=numpy.str_)
        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if tag != "replace" and tag!="equal":
                return [],None
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
            text_edits.append((tag, text[i1:i2], i1, i2, corrected_text[j1:j2], j1, j2))

            word_ids=self.tokenizer.convert_tokens_to_ids(list(corrected_text[j1:j2]))
            slice_start,slice_end=self.findSliceWordCandidates(corrected_text,i1,corrected_text[i1:i2],correct_top1_ids,word_ids)
            candidates = numpy.append(candidates, text_word_recalls[slice_start:slice_end], axis=0)
        return text_edits, candidates

    def isSemanticReplace(self,err_word_pinyin_candidates,err_word_recalltopk_candidates,src_err_words):
        # src_err_words格式：[(tag, 待替换字, i1, i2, 替换字, j1, j2), (...),,, ]
        # 针对模型语义替换，且召回集中有与原字同音形的，则再次纠错
        for s_word_tuple in src_err_words:
            if len(s_word_tuple[1])!=len(s_word_tuple[4]):
                return False
            s_pinyins,r_pinyins=[],[]
            for index,err_word in enumerate(s_word_tuple[1]):
                s_err_word_pinys=lazy_pinyin(err_word,Style.NORMAL)
                r_word=s_word_tuple[4][index]
                r_word_pinys=lazy_pinyin(r_word,Style.NORMAL)
                r_sim_pinys=[]
                r_sim_pinys.extend(r_word_pinys)
                for r_py in r_word_pinys:
                    sim_pinys=self.pyUtil.recoverySimPinyinFromCore(r_py,contains_diff_tone=False)
                    r_sim_pinys.extend(sim_pinys)
                for s_py in s_err_word_pinys:
                    if s_py not in r_sim_pinys:
                        # 跨语义预测
                        return True
        return False
    def getFineSimChinese(self,text,errWord,pos):
        simChineses = self.pyUtil.getSimilarityChineseBySimPinyin(errWord)
        # 同音字中排除非本句读音的多音同音汉字
        fineSimChineses = self.filterSimChineseByCurPinyin(simChineses, text, errWord)
        return fineSimChineses
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
            m1_edits.append((tag, src_text[i1:i2], m1_text[j1:j2], i1,i2,j1,j2))
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

    def findCandidateWordsPerErrPos(self, src_text, src_err_words, topkRecallPerErrPos,thresh=0.8):
        candidate_confusions = self.getConfusionsPerErrWords(src_err_words)
        candidate_pin_shapes = self.getPinShapePerErrWords(src_text,src_err_words,thresh=thresh)
        candidate_pin_shapes_from_topk = self.filterByPinShape(src_text,src_err_words,topkRecallPerErrPos,thresh=thresh)
        # print(src_err_words,candidate_pin_shapes,candidate_pin_shapes_from_topk)

        return candidate_pin_shapes,candidate_pin_shapes_from_topk
    def getConfusionsPerErrWords(self, src_err_words):
        # todo
        return None

    def getPinShapePerErrWords(self, src_text, src_err_words,thresh=0.8,scorePinyFactor=1.2,scoreShapeFactor=1.1):
        error_word_candidates=[]
        for wordTuple in src_err_words:
            for index,wordErr in enumerate(wordTuple[1]):
                simChineses=self.pyUtil.getSimilarityChineseBySimPinyin(wordErr)
                # 同音字中排除非本句读音的多音同音汉字
                fineSimChineses=self.filterSimChineseByCurPinyin(simChineses,src_text,wordErr,scorePinyFactor=scorePinyFactor)
                fineSimChineses.extend([(simShapeWord,scoreShapeFactor)
                                        for simShapeWord in self.shapeUtil.getAllSimilarityShape(wordErr,thresh=thresh)])
                # if matchGroup:
                # 逐一判断是否在原句组成词组，非组成词组的权重分低
                candidateSimWords = []
                for simWord_tuple in fineSimChineses:
                    replace_text=src_text[:wordTuple[2]+index]+simWord_tuple[0]+src_text[wordTuple[2]+index+1:]
                    pos=wordTuple[2]+index
                    left=max(pos-5,0)
                    right=min(len(replace_text),pos+5)
                    splits_words=self.thu.cut(replace_text[left:right],cut_all=False) #加快分词速度
                    size=0
                    offset=left
                    if left!=0:
                        offset=left-1
                    for s_words in splits_words:
                        size+=len(s_words[0])
                        if size<=pos-offset:
                            continue
                        if simWord_tuple[0] in s_words[0]  and len(s_words[0])>1:
                            candidateSimWords.append((simWord_tuple[0], simWord_tuple[1]))
                        else:
                            candidateSimWords.append((simWord_tuple[0], 0.9))
                        break
                error_word_candidates.append((wordTuple[1][index],wordTuple[2]+index,set(candidateSimWords)))
                # else:
                #     error_word_candidates.append((wordTuple[1][index],wordTuple[2]+index,fineSimChineses))
        return error_word_candidates
    def filterByPinShape(self,src_text, src_err_words, topkRecallPerErrPos,thresh=0.8):
        simChineses=[]
        for wordTuple in src_err_words:
            for index, wordErr in enumerate(wordTuple[1]):
                simPinChineses = self.pyUtil.getSimilarityChineseBySimPinyin(wordErr)
                # 同音字中排除非本句读音的多音同音汉字
                fineSimChineses = self.filterSimChineseByCurPinyin(simPinChineses, src_text, wordErr)
                simShapeChineses= self.shapeUtil.getAllSimilarityShape(wordErr,thresh=thresh)
                simChineses.append(fineSimChineses)
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

    def computeReplaceScore(self, text,err_word_pinyin_candidates, err_word_recalltopk_candidates):
        if err_word_pinyin_candidates==None:
            return None,[]
        word_scores={}

        for tuple3 in err_word_pinyin_candidates:
            pos=tuple3[1]
            candidate_words=tuple3[2]
            r_word_scores={}
            record_s_score = {}
            for r_word_tuple2 in candidate_words:
                new_text=text[:pos]+r_word_tuple2[0]+text[pos+1:]
                flag,r_score,s_score=self.word2vecSim.doReplace(text,new_text)
                r_word_scores[r_word_tuple2[0]]=r_score * r_word_tuple2[1]
                record_s_score[tuple3[0]] = s_score
            word_scores[tuple3[0]]={**r_word_scores, **record_s_score}

        for index,word_candidates in enumerate(err_word_recalltopk_candidates):
            pos=err_word_pinyin_candidates[index][1]
            src_word=err_word_pinyin_candidates[index][0]
            r_word_scores={}
            s_word_scores={}
            if src_word in word_scores:
                r_word_scores=word_scores[src_word]
            for r_word in word_candidates:
                new_text = text[:pos] + r_word + text[pos + 1:]
                # 计算单词局部语义得分
                flag, r_score, s_score = self.word2vecSim.doReplace(text, new_text)
                r_word_scores[r_word] = r_score
                # 判断topk候选集中是否存在形近音近，存在则优先选择
                sim_candidates=err_word_pinyin_candidates[index][2]
                if r_word in sim_candidates:
                    r_word_scores[r_word]=r_word_scores[r_word] * 2
            word_scores[src_word] = r_word_scores
        sorted_scores={}
        for key,values in word_scores.items():
            sorted_similarity = sorted(values.items(), key=itemgetter(1), reverse=True)
            sorted_scores[key]=sorted_similarity
        return sorted_scores

    def chooseBestCandidate(self, text,err_word_pinyin_candidates, scores):
        new_text=text
        # if len(firstCandidates)>0:
        #     for w_t3 in firstCandidates:
        #         new_text=new_text[:w_t3[1]]+w_t3[2]+new_text[w_t3[1]+1:]
        #     return new_text,firstCandidates

        # err_word_pinyin_candidates格式示例：[('诉', 43, ['塑', '宿',,,),(...),]，其中’诉‘为原字，43：为原字位置，列表中为候选字
        # scores字典格式示例：{'诉': [('溯', 0.1488705426454544), ('斥', 0.1488705426454544),,,]}
        # print(err_word_pinyin_candidates,scores)
        for word_candidates in err_word_pinyin_candidates:
            if len(scores[word_candidates[0]])==0:
                continue
            new_text=new_text[:word_candidates[1]]+scores[word_candidates[0]][0][0]+new_text[word_candidates[1]+1:]
        return new_text,scores

    def split_by_douhao(self, text):
        texts=text.split(sep='，')
        target_size=0
        text_list=[]
        fine_str=''
        for sub_text in texts:
            target_size+=len(sub_text)
            if target_size<128:
                fine_str+=sub_text
            else:
                text_list.append(fine_str)

                target_size=len(sub_text)
                fine_str=sub_text
        text_list.append(fine_str)
        return text_list

    def filterSimChineseByCurPinyin(self, simChineses, src_text, wordErr, contains_diff_tone=False, scorePinyFactor=1.2):
        # 为方便定位，首先过滤非汉字逗号字符（尤其排除英文：其会作为整体占一位）
        fine_src_text=''
        new_pos=0
        i=0
        find_flag=False
        for sw in src_text:
            if isChinese(sw)==False and sw!='，':
                continue
            fine_src_text += sw
            if sw==wordErr and find_flag==False:
                new_pos=i
                find_flag=True
            if find_flag==False:
                i+=1

        src_pinyins=pinyin(fine_src_text,style=Style.TONE3)
        word_pinyins=src_pinyins[new_pos]
        # 扩增近音
        temp_sim_py=[]
        for word_py in word_pinyins:
            sim_word_pinyins=self.pyUtil.recoverySimPinyinFromCore(word_py,contains_diff_tone=contains_diff_tone)
            temp_sim_py.extend(sim_word_pinyins)
        word_pinyins.extend(temp_sim_py)
        word_pinyins=set(word_pinyins)

        # simChineses中汉字逐一替换到原文获取正确读音
        fine_sim_chineses=[]
        for word in simChineses:
            new_text=fine_src_text[:new_pos]+word+fine_src_text[new_pos+1:]
            new_pinyins=pinyin(new_text,style=Style.TONE3)
            actual_pinyins=new_pinyins[new_pos]
            # 是否存在于word_pinyins中
            flag=False
            for a_py in actual_pinyins:
                if a_py in word_pinyins:
                    flag=True
                    break
            if flag==True:
                fine_sim_chineses.append((word,scorePinyFactor))
        return fine_sim_chineses

    def computeTopk(self,correct_top1_ids, topk_texts, val_target):
        tuple7_list = self.getTwoTextEdits(correct_top1_ids, val_target)
        indexs=[]
        for tuple7 in tuple7_list:
            slices=topk_texts[tuple7[3]:tuple7[4],:]
            target_slice=val_target[tuple7[5]:tuple7[6]]
            flag=False
            for j,topi in enumerate(slices):
                if topi in target_slice:
                    flag=True
                    break
            if flag==False:
                return False,-1
            else:
                indexs.append((tuple7[3],tuple7[4]))
        return True,indexs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--macbert_model_dir", default='pretrained/macbert4csc',
                        type=str,
                        help="MacBert pre-trained model dir")
    args = parser.parse_args()

    m = MacBertCorrector(args.macbert_model_dir)
    error_sentences = [
        '演员建立在厚重剧作基础之上的出色表演为影平添彩。'
        # '#文明礼仪微讲堂#（四）四、公务礼仪一当面接待扎仪上级来访，接待要周到。'
    ]
    target_sentences=[
        # '#文明礼仪微讲堂#（四）四、公务礼仪一当面接待礼仪上级来访，接待要周到。'
        '演员建立在厚重剧作基础之上的出色表演为影片添彩。'
    ]
    t1 = time.time()
    for sent in error_sentences:
        corrected_sent, scores, err = m.macbert_correct_recall(sent,val_target=target_sentences[0])
        print("original sentence:{} => {} err:{}".format(sent, corrected_sent, err))
    print('[single]spend time:', time.time() - t1)
    t2 = time.time()
    res = m.batch_macbert_correct(error_sentences)
    for sent, r in zip(error_sentences, res):
        print("original sentence:{} => {} err:{}".format(sent, r[0], r[1]))
    print('[batch]spend time:', time.time() - t2)
