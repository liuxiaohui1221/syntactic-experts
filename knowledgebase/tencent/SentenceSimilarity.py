import os.path
from difflib import SequenceMatcher
from imp import reload

import model.model_MiduCTC.src.thulac as thulac
import gensim

from ProjectPath import get_project_path
from data_augmentation.preliminary_gen import isChinese

class WordSentenceSimliarity:
    def __init__(self):
        self.wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(
            os.path.join(get_project_path(),'knowledgebase/tencent/tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'),binary=False)
        self.wv_from_text.fill_norms()
        # 分词
        self.thu1 = thulac.thulac(seg_only=True)  #只进行分词，不进行词性标注


    def findPreKeyWord(self,words, word, skipEnds):
        skip=0
        findFirst=0
        curWord=word
        pre_word=None
        for i in range(len(words)-1,-1,-1):
            split_word=words[i][0]
            skip+=len(split_word)
            if skip<skipEnds:
                continue
            # 开始判断合法前关键字
            isvalid = self.isValidKeyWord(split_word, word)
            if isvalid == False:
                if word==split_word:
                    # first in is curWord
                    curWord = split_word
                    findFirst=1
                else:
                    continue
            if findFirst==0:
                # first in is curWord
                curWord=split_word
                findFirst=1
            if findFirst==2:
                pre_word=split_word
                break
            if findFirst==1:
                findFirst=2
        return curWord,pre_word

    def findRearKeyWord(self,words, word, skipsStarts):
        skip = 0
        findFirst = 0
        curWord = None
        rearWord = None
        for i in range(len(words)):
            split_word = words[i][0]
            skip += len(split_word)
            if skip < skipsStarts:
                continue
            # 开始判断合法前关键字
            isvalid = self.isValidKeyWord(split_word, word)
            if isvalid == False:
                if word == split_word:
                    # first in is curWord
                    curWord = split_word
                    findFirst = 1
                else:
                    continue
            if findFirst == 0:
                # first in is curWord
                curWord = split_word
                findFirst = 1
            if findFirst == 2:
                rearWord = split_word
                break
            if findFirst == 1:
                findFirst = 2
        return curWord,rearWord
    def isValidKeyWord(self,readyword,matchingWord):
        # 排除相等词
        if readyword == matchingWord:
            return False
        # 排除停用词及不在词向量表中的词
        if self.wv_from_text.get_index(readyword, -1) == -1:
            return False
        invalid = True
        for w in readyword:
            if isChinese(w) == False:
                invalid = False
                break
        return invalid
    def _getRelatedKeyWords(self,sentence,matchingWord,start,end):
        # 前后相关词:
        pos,pre_pos,rear_pos=-1,-1,-1
        words = self.thu1.cut(sentence)
        for index,word in enumerate(words):
            word_str=word[0]
            # 合法关键词
            isvalid = self.isValidKeyWord(word_str, matchingWord)
            if isvalid == False:
                if matchingWord==word_str:
                    pos=index
                    break
                continue
            if matchingWord in word_str:
                pos=index
                break
            pre_pos=index
        if pos==-1:
            # 分词器将matchingWord分开了，若分开的字与其他字组成词组则将次作为前或后关键字
            matchingWords = self.thu1.cut(matchingWord)
            curWord1,preWord=self.findPreKeyWord(words,matchingWords[0][0],len(sentence)-start)
            curWord2,rearWord=self.findRearKeyWord(words,matchingWords[-1][0],start+(len(matchingWord)))
            keyWords = []
            if preWord != None:
                keyWords.append(preWord)
            if rearWord != None:
                keyWords.append(rearWord)
            return [curWord1,curWord2],keyWords
        else:
            # 分词器没有把matchingWord分开的场景：
            # 从pos位置开始查找后相关词
            if pos != -1 and pos!=len(words)-1:  # 可能不在词典中被排除
                for index,word in enumerate(words[pos+1:]):
                    word_str = word[0]
                    # 合法关键词
                    isvalid=self.isValidKeyWord(word_str,words[pos])
                    if isvalid==False:
                        continue
                    rear_pos=index+pos+1
                    break
        preWord,curWord,rearWord=None,None,None
        if pos!=-1:# 可能不在词典中被排除
            curWord=words[pos][0]
        if pre_pos!=-1:
            preWord=words[pre_pos][0]
        if rear_pos!=-1:
            rearWord=words[rear_pos][0]

        keyWords=[]
        if preWord!=None:
            keyWords.append(preWord)
        if rearWord!=None:
            keyWords.append(rearWord)
        return [curWord],keyWords


    def computeWordSimilarity(self,source,reference,sword,tword,s_start,s_end,t_start,t_end):
        curWords1,relatedKeyWords1 =self._getRelatedKeyWords(source,sword,s_start,s_end)
        if len(relatedKeyWords1)==0 or len(curWords1)==0:
            return 1,1
        curWords2,relatedKeyWords2 =self._getRelatedKeyWords(reference,tword,t_start,t_end)
        if len(relatedKeyWords2)==0 or len(curWords2)==0:
            return 1,1

        # print(curWords1, relatedKeyWords1)
        # print(curWords2, relatedKeyWords2)
        sim1=self.wv_from_text.n_similarity(curWords1,relatedKeyWords1)
        sim2=self.wv_from_text.n_similarity(curWords2,relatedKeyWords2)
        # print('相似度：',sim1,sim2)
        score1=sim1*(1/(len(curWords1)+len(relatedKeyWords1))*len("".join(relatedKeyWords1)))*0.1
        score2=sim2*(1/len(curWords2)+len(relatedKeyWords2)*len("".join(relatedKeyWords2)))*0.1
        # print(source, reference, sword, tword, score1, score2)
        return score1,score2
    def getSpellErrorWord(self,source,target):
        r = SequenceMatcher(None, source, target)
        diffs = r.get_opcodes()
        s_words = []
        t_words = []
        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if tag == 'replace':
                s_words.append((source[i1:i2],i1,i2))
                t_words.append((target[j1:j2],j1,j2))
        return s_words,t_words


    def checkAndGetCoreWrodsInDB(self,s_word):
        s_core_words=[]
        if self.wv_from_text.get_index(s_word, -1) == -1:
            s_words=self.thu1.cut(s_word)
            for w_tuple in s_words:
                if self.wv_from_text.get_index(s_word, -1) == -1:
                    continue
                s_core_words.append(w_tuple[0])
        else:
            s_core_words.append(s_word)
        return s_core_words

    def doReplace(self,source,reference,thresh=0.15):
        s_words,t_words=self.getSpellErrorWord(source, reference)
        if len(s_words)==0:
            return False,1
        # 若不存在词典中则分词，若两个分词列表不等长，相似度按长度均分权重
        s_score,t_score=0,0
        s_weight=1.0/len(s_words)
        for index,s_w in enumerate(s_words):
            # 若不存在词典中则分词，若两个分词列表不等长，相似度按长度均分权重
            ss_words = self.checkAndGetCoreWrodsInDB(s_w[0])
            tt_words = self.checkAndGetCoreWrodsInDB(t_words[index][0])
            if len(ss_words)==0 or len(tt_words)==0:
                continue
            # 不等长问题
            ss_weight=1.0/len(ss_words)
            tt_weight=1.0/len(tt_words)
            ss_score,tt_score=0,0
            for k,match_word1 in enumerate(ss_words):
                score1,score2=self.computeWordSimilarity(source, reference, match_word1, tt_words[k],s_w[1],s_w[2],t_words[index][1],t_words[index][2])
                ss_score+=ss_weight*score1
                tt_score+=tt_weight*score2
            s_score+=s_weight*ss_score
            t_score+=tt_weight*tt_score
        # print("s_score,t_score:",s_score,t_score)
        if s_score-t_score>=thresh:
            # 拒绝替换
            return False,t_score
        return True,t_score
if __name__ == "__main__":
    # target = "这样才有空间塞燕窝"
    texts1 = "造成粮食欠收。"
    m1 = "造成粮食歉收。"
    m2 = "造成粮食欠收。"
    wss=WordSentenceSimliarity()
    isReplace = wss.doReplace(texts1, m1)
    isReplace2 = wss.doReplace(texts1, m2)
    print("句1：", texts1)
    print("m1：", m1,isReplace)
    print("m2：", m2, isReplace2)
    # s_words,t_words=getSpellErrorWord(texts1,texts2)



