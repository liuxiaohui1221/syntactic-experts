import os.path
from difflib import SequenceMatcher

from gensim.models import KeyedVectors

import models.model_MiduCTC.src.thulac as thulac
import gensim

from ProjectPath import get_project_path
from data_augmentation.preliminary_gen import isChinese
from models.macbert.util.common import filterNonChinese
from models.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf


class WordSentenceSimliarity:
    def __init__(self,thulac_singleton=None):
        word2vec_model_path_txt=os.path.join(get_project_path(),'knowledgebase/tencent/tencent-ailab-embedding-zh-d200-v0.2.0/tencent-ailab-embedding-zh-d200-v0.2.0.txt')
        word2vec_model_path_mmap=word2vec_model_path_txt.replace(".txt",".bin")

        # self.wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path_txt)
        self.wv_from_text = gensim.models.KeyedVectors.load(word2vec_model_path_mmap, mmap='r')
        self.wv_from_text.fill_norms()
        # self.wv_from_text.save(word2vec_model_path_mmap)
        # 分词
        # self.thu1 = VocabConf().jieba_singleton
        self.thu1 = VocabConf().thulac_singleton

    def existTencentWord(self,readyword):
        num=self.wv_from_text.get_index(readyword, -1)
        if num == -1:
            return False
        return True

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
        curWord = ''
        rearWord = ''
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
        # words = self.thu1.cut(sentence,cut_all=False)
        words = self.thu1.cut(sentence)
        invalidSplitStr=None
        skip=0
        for index,word in enumerate(words):
            word=word[0]
            word_str=word
            skip+=len(word_str)
            isvalid = self.isValidKeyWord(word_str, matchingWord)
            if skip<=start:
                if isvalid:
                    pre_pos = index
                continue
            # 合法关键词
            if isvalid == False:
                if matchingWord==word_str:
                    pos=index
                    break
                if matchingWord in word_str:
                    pos=index
                    invalidSplitStr=word_str
                    break
                continue
            if matchingWord in word_str:
                pos=index
                break

        if pos==-1:
            # 分词器将matchingWord分开了，若分开的字与其他字组成词组则将次作为前或后关键字
            # matchingWords = self.thu1.lcut(matchingWord,cut_all=False)
            matchingWords = self.thu1.cut(matchingWord)
            curWord1,preWord=self.findPreKeyWord(words,matchingWords[0][0],len(sentence)-start)
            curWord2,rearWord=self.findRearKeyWord(words,matchingWords[-1][0],start+(len(matchingWord)))
            keyWords = []
            if preWord != None:
                keyWords.append(preWord)
            if rearWord != None:
                keyWords.append(rearWord)
            if curWord1 == None or curWord2 == None:
                return None, keyWords
            return [curWord1,curWord2],keyWords
        else:
            if invalidSplitStr!=None and len(matchingWord)==1:
                pre_pos=-1
                find_flag=0
                for index,word in enumerate(sentence):
                    if matchingWord == word:
                        pos=index
                        find_flag=1
                    if find_flag==1 and self.isValidKeyWord(word, matchingWord):
                        rear_pos=index
                        break
                    if find_flag==0:
                        pre_pos=index
                return [sentence[pos]],[sentence[pre_pos],sentence[rear_pos]]
            else:
                # 分词器没有把matchingWord分开的场景：
                # 从pos位置开始查找后相关词
                if pos != -1 and pos!=len(words)-1:  # 可能不在词典中被排除
                    for index,word in enumerate(words[pos+1:]):
                        word_str = word[0]
                        # 合法关键词
                        isvalid=self.isValidKeyWord(word_str,words[pos][0])
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
        if curWord==None:
            return None,keyWords
        return [curWord],keyWords
    def computeSimilarity(self,word1,word2,type="word"):
        # 过滤非汉字
        filtered_word1=filterNonChinese(word1)
        filtered_word2 = filterNonChinese(word2)
        if len(filtered_word1)==0 or len(filtered_word2)==0:
            return 0
        if type=="word":
            return self.wv_from_text.n_similarity(filtered_word1,filtered_word2)
        else:
            # 句子中不同部分与相同部分的相似度比较
            pass

    # def computeWordSimilarity(self,source,reference,sword,tword,s_start,s_end,t_start,t_end):
    #     curWords1,relatedKeyWords1 =self._getRelatedKeyWords(source,sword,s_start,s_end)
    #     if len(relatedKeyWords1)==0 or curWords1==None:
    #         return 1,1
    #     curWords2,relatedKeyWords2 =self._getRelatedKeyWords(reference,tword,t_start,t_end)
    #     if len(relatedKeyWords2)==0 or curWords2==None:
    #         return 1,1
    #     try:
    #         sim1=self.wv_from_text.n_similarity(curWords1,relatedKeyWords1)
    #         sim2=self.wv_from_text.n_similarity(curWords2,relatedKeyWords2)
    #         # curWords与句子的相似度：curWords被切分的词组越少得分越高，分词长度保留越长得分越高，curWords依赖的前后词越长，算出的得分权重越大
    #         score1=sim1 * (1/(len(curWords1)) * (len("".join(curWords1))))
    #         score2=sim2 * (1/(len(curWords2)) * (len("".join(curWords2))))
    #     except KeyError as e:
    #         print(e)
    #         score1=score2=0
    #     print("与前后词相似度：",(sim1,round(score1, 4)),curWords1, relatedKeyWords1)
    #     print("与前后词相似度：",(sim2,round(score2, 4)),curWords2, relatedKeyWords2)
    #     return round(score1, 4),round(score2, 4)


    def checkAndGetCoreWrodsInDB(self,s_word):
        s_core_words=[]
        if self.wv_from_text.get_index(s_word, -1) == -1:
            # s_words=self.thu1.lcut(s_word,cut_all=False)
            s_words = self.thu1.cut(s_word)
            for w_tuple in s_words:
                w_tuple=w_tuple[0]
                if self.wv_from_text.get_index(s_word, -1) == -1:
                    continue
                s_core_words.append(w_tuple)
        else:
            s_core_words.append(s_word)
        return s_core_words

    # def doReplace(self,source,reference,thresh=0.001):
    #     s_words,t_words=self.getSpellErrorWord(source, reference)
    #     if len(s_words)==0:
    #         # 无纠错
    #         return None,-1,-1
    #     # 若不存在词典中则分词，若两个分词列表不等长，相似度按长度均分权重
    #     s_score,t_score=0,0
    #     s_weight=1.0/len(s_words)
    #     for index,s_w in enumerate(s_words):
    #         # 若不存在词典中则分词，若两个分词列表不等长，相似度按长度均分权重
    #         ss_words = self.checkAndGetCoreWrodsInDB(s_w[0])
    #         tt_words = self.checkAndGetCoreWrodsInDB(t_words[index][0])
    #         if len(ss_words)==0 or len(tt_words)==0:
    #             continue
    #         # 不等长问题
    #         ss_weight=1.0/len(ss_words)
    #         tt_weight=1.0/len(tt_words)
    #         ss_score,tt_score=0,0
    #         for k,match_word1 in enumerate(ss_words):
    #             score1,score2=self.computeWordSimilarity(source, reference, match_word1, tt_words[k],s_w[1],s_w[2],t_words[index][1],t_words[index][2])
    #             ss_score+=ss_weight*score1
    #             tt_score+=tt_weight*score2
    #         s_score+=s_weight*ss_score
    #         t_score+=tt_weight*tt_score
    #
    #     print("与前后词相似度：",s_score,s_words)
    #     print("与前后词相似度：",t_score,t_words)
    #     if s_score-t_score>thresh:
    #         # 拒绝替换
    #         return False,t_score,s_score
    #     return True,t_score,s_score
if __name__ == "__main__":
    # target = "这样才有空间塞燕窝"
    texts1 = [
        "几乎翻了一倍",
        "非洲优先和世界遗产之间的关係",
        "造成粮食欠收。",
        "请有关单位和人员作好防...http:t.cnh1ZU5​​",
        "#夜聊#小豆芽们的期末考试都考的怎么样呢？",
        "积极吸纳发展35岁以下积极份子和年轻党员，解决党员老龄化问题。"
        "#悦读麦积##数字阅读推荐#【#麦图讲座#（第399期）|漫谈银幕上的女性党员形象】#读书#从《烈火中永生》的“江姐”到《风声》的“老枪”，百年影史里留下了众多鲜活的女性党员形象，她们大多在花样年华舍身取义，她们拥有爱情、家人、同时也献身革命，在家国叙事中完成了自我成长。",
        "在此之前，它们始终蜇伏在地图上，如今,它们就要向我花枝招展了。",
        "真正的相濡以沫是怎样的#爱是罗曼蒂克，爱是细水流长，爱也是柴米油盐。",
        "正直炎热的周末应该在家喝着冰凉可口的饮品吃着西瓜刷着抖音……好不惬意！",
        "清理2处紧邻居民楼陡坡上的杂树，督促部分工程停止施工、封闭围档、清理场地。",
        "防御指南:1.政府及相关部门按照职责做好防短时暴雨、防雷、防大风准备工作，气象部门做好人工防雹作业准备；2.户外行人和工作人员减少户外活动，注意远离棚架广告牌等搭建物；"
        "3.驱赶家禽、牲畜进入有顶蓬的场所，关好门窗加固棚舍；4.检查城市、农田、鱼塘排水系统，做好排涝准备和对山洪、滑坡、泥石流等灾害的防御准备。"
    ]
    correct_text = [
        "几乎翻了一培",
        "非洲优先和世界遗产之间的关系",
        "造成粮食歉收。",
        "请有关单位和人员做好防...http:t.cnh1ZU5​​",
        "#夜聊#小豆芽们的期末考试都考得怎么样呢？",
        "积极吸纳发展35岁以下积极分子和年轻党员，解决党员老龄化问题。",
        "#悦读麦积##数字阅读推荐#【#麦图讲座#（第399期）|漫谈银幕上的女性党员形象】#读书#从《烈火中永生》的“江姐”到《风声》的“老枪”，百年影史里留下了众多鲜活的女性党员形象，她们大多在花样年华舍生取义，她们拥有爱情、家人、同时也献身革命，在家国叙事中完成了自我成长。",
        "在此之前，它们始终蛰伏在地图上，如今,它们就要向我花枝招展了。",
        "真正的相濡一沫是怎样的#爱是罗曼蒂克，爱是细水流长，爱也是柴米油盐。",
        "正值炎热的周末应该在家喝着冰凉可口的饮品吃着西瓜刷着抖音……好不惬意！",
        "清理2处紧邻居民楼陡坡上的杂树，督促部分工程停止施工、封闭围挡、清理场地。",
        "防御指南:1.政府及相关部门按照职责做好防短时暴雨、防雷、防大风准备工作，气象部门做好人工防雹作业准备；2.户外行人和工作人员减少户外活动，注意远离棚架广告牌等搭建物；"
        "3.驱赶家禽、牲畜进入有顶篷的场所，关好门窗加固棚舍；4.检查城市、农田、鱼塘排水系统，做好排涝准备和对山洪、滑坡、泥石流等灾害的防御准备。"
    ]
    # m2 = "造成粮食欠收。"
    wss=WordSentenceSimliarity()
    # for index,text in enumerate(texts1):
    #     isReplace = wss.doReplace(text, correct_text[index])
    #     # 句1： 造成粮食欠收。
    #     # m1： 造成粮食歉收。 (True, 0.2091964602470398, 0.05585148334503174)
    #     # m2： 造成粮食欠收。 (False, -1, -1)
    #     print("[(replace，keep), src_text]：", isReplace,text)
    print("*"*50)
    scores=wss.computeSimilarity("几乎翻了一","倍")
    scores2=wss.computeSimilarity("几乎翻了一","培")
    print(scores,scores2)

    scores1 = wss.computeSimilarity("先你喜欢的事。", "情")
    scores2 = wss.computeSimilarity("先你喜欢的事情。", "做")
    print(scores1, scores2)

    score1 = wss.computeSimilarity('人都接种', '人痘接种')
    print(score1)



