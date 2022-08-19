from difflib import SequenceMatcher
from operator import itemgetter

import numpy

from models.mypycorrector.utils.text_utils import is_chinese


def stopDuplicateCheck(w,word):
    stopchecks=['队','军','省','市','县','村','每','榜','年','图片','妈妈','由','丝']
    stopwords=['不着急','丝丝']
    if w in stopchecks or word in stopwords:
        return True
    return False

def removeDuplicate(fenci, text):
    arr=fenci.lcut(text)
    # 相邻存在包含关系的
    pre=None
    pre_index=-1
    fine_text = numpy.array(arr)
    for index,word in enumerate(arr):
        fine_text[index] = word
        flag=False
        for w in word:
            if is_chinese(w)==False and stopDuplicateCheck(w,word):
                flag=True
                break
        if flag==False:
            if pre and (len(pre)>1 or len(word)>1):
                if pre and len(word)>len(pre) and word[:len(pre)]==pre:
                    # del pre
                    print("del word:", fine_text[pre_index],"from:",text)
                    fine_text[pre_index]=''
                elif pre and len(pre)>=len(word) and pre[len(pre)-len(word):]==word:
                    # del cur word
                    print("del word:",fine_text[index],"from:",text)
                    fine_text[index]=''
        pre=word
        pre_index=index
    return "".join(fine_text)

def fenciCorrect(ltp, src_text, final_corrected):
    # 1.模型预测文本再次分词，并比较前后两次分词结果：分词后词组数变长的忽略修改
    src_output = ltp.pipeline(src_text, tasks=["cws", "pos", "ner", "srl", "dep", "sdp"])
    corr_output = ltp.pipeline(final_corrected, tasks=["cws", "pos", "ner", "srl", "dep", "sdp"])
    src_arr = src_output.cws
    corr_arr = corr_output.cws
    if len(corr_arr) > len(src_arr):
        final_corrected = src_text
    return final_corrected

def getTextEdits(src_text, m1_text):
    if m1_text==None:
        return None
    r = SequenceMatcher(None, src_text, m1_text)
    diffs = r.get_opcodes()
    m1_edits = []
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        if "equal" in tag:
            continue
        m1_edits.append(diff)
    return m1_edits
def getEdits(src_text, m1_text):
    if m1_text==None:
        return None
    r = SequenceMatcher(None, src_text, m1_text)
    diffs = r.get_opcodes()
    m1_edits = []
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        if "equal" in tag:
            continue
        m1_edits.append((tag,src_text[i1:i2],m1_text[j1:j2]))
    return m1_edits
def getSpellErrorWord(source,target):
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

def filterUpdateOtherProper(ltp,src_text,final_corrected,stop_tokens=['nh']):
    edits=getTextEdits(src_text,final_corrected)
    output = ltp.pipeline(src_text, tasks=["cws", "pos", "ner", "srl", "dep", "sdp"])
    word_arr=output.cws
    ops_arr=output.pos
    # 定位不可修改索引范围列表
    index=0
    stop_range=[]
    for i,word in enumerate(word_arr):
        if ops_arr[i] in stop_tokens:
            stop_range.append((index,index+len(word)))
        index+=len(word)
    # 排除不可编辑位置
    for edit in edits:
        src_edit_range=(edit[1],edit[2])
        corr_edit_range=(edit[3],edit[4])
        for s_range in stop_range:
            if (src_edit_range[0]<s_range[1] and src_edit_range[1]>s_range[0]):
                new_correct=final_corrected[:corr_edit_range[0]]+src_text[src_edit_range[0]:src_edit_range[1]]+final_corrected[corr_edit_range[1]:]
                # 一般只有一次纠改，故跳出
                print("stop edit range:",final_corrected[corr_edit_range[0]:corr_edit_range[1]],src_text[src_edit_range[0]:src_edit_range[1]],new_correct)
                return new_correct
    return final_corrected


def filterNonChinese(text):
    chinese_text=[]
    for word in text:
        if is_chinese(word)==False:
            continue
        chinese_text.append(word)
    return "".join(chinese_text)
def getRecallCorrected(wss, text, edits,threshold=0.01):
    candidates_correcteds={}
    dtails=[]
    for t in edits:
        text_new = text[:t[2]] + t[1] + text[t[3]:]
        temp_trunc = filterNonChinese(text[:t[2]]+text[t[3]:])
        try:
            temp1 = filterNonChinese(t[0])
            temp2 = filterNonChinese(t[1])
            if len(temp1)<1 or len(temp2)<1:
                continue
            keep_score = wss.computeSimilarity(temp_trunc, temp1)
            replace_score = wss.computeSimilarity(temp_trunc, temp2)
        except KeyError as e:
            print(e)
            continue
        if replace_score-keep_score<threshold:
            continue
        candidates_correcteds[text_new]=replace_score-keep_score
        dtails.append(t)
    return candidates_correcteds,dtails


def chooseBestCorrectCandidate(wss,text,corrected_edits,target=None,threshold=0.01,topN=10):
    filtered_texts, details = getRecallCorrected(wss,text, corrected_edits,threshold=threshold)
    # 从候选纠错集中计算与原句得分：
    final_text = text
    recalled=False
    if target in filtered_texts:
        recalled=True
    # 选择得分最高的文本
    print(filtered_texts,details)
    sorted_similarity = sorted(filtered_texts.items(), key=itemgetter(1), reverse=True)
    topSimShapeChineses = [chars for chars, similarity in sorted_similarity[:topN]]
    return topSimShapeChineses,details,recalled

def filterSemanticChanged():
    # if final_corrected != src_text:
    #     isReplace = wss.doReplace(src_text, final_corrected)
    #     if isReplace == False:
    #         final_corrected = src_text
    pass