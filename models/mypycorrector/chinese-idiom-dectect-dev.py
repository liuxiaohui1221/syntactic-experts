# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 自定义成语纠错
"""
import json
import os.path
from difflib import SequenceMatcher

from tqdm import tqdm

from ProjectPath import get_project_path
from models.mypycorrector.utils.string_util import getTwoTextEdits

from models.mypycorrector.corrector import Corrector

def containKeyEdits(reference_list,src_text, trg_text):
    r = SequenceMatcher(None, src_text, trg_text)
    diffs = r.get_opcodes()
    m1_edits = []
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        if "equal" in tag:
            continue
        if "replace" !=tag:
            return False
        flag=False
        for ref_tuple in reference_list:
            if ref_tuple[0]==src_text[i1:i2] and ref_tuple[1]==trg_text[j1:j2]:
                flag=True
        if flag==False:
            return False
    return True

if __name__ == '__main__':
    error_sentences = [
        "警方提醒请勿利令智",
        "杜飞对如萍的爱近乎痴情，整部剧中，随时随地为如萍鞍前马不说。",
        "以城为镜，弘扬中华民主传统美德，让我们一起用实际行动为信阳的文明发展贡献出自己的一份力量！",
    "警方当下立断，让其父亲将梯子架在屋檐旁，并在梯子底下埋伏一个突击小组。",
    "2021年8月9日傍晚6时左右，虹桥镇万源新城3期117号门口的非机动车车棚里一辆电动自行车充电时突然自燃，车内的电瓶发生爆炸，火势一簇而上，又因当天风力强劲，火势迅速向西蔓延。",
    "自禄口机场暴发疫情以来，潘金海积极响应号召，招募志愿者，成立美年大健康志愿者服务小分队，组织防疫培训、帮忙搭建遮阳棚、协助核酸采集……每天忙得马不停歇。",
    "走进县直机关入党积极分子暨发展对象培训班，和瑞庭同志用自己的亲生经历为这92名入党积极分子上了一堂深刻的爱国主义教育课。",
        "要重点关注城市配套费、农民讨薪、国企改革、房地产遗留等问题，强化督办考核，以更优更好的举措、更严更实的态度，办好一件建议提案、解决好一类问题、推动一方面工作、造福一方百姓。",
        "沪水域市容环境卫生管理规定开始施行 《上海市水域市容环境卫生管理规定》共26条，是对现行政府规章《上海市水域环境卫生管理规定》的废旧立新，进一步明晰部门职责、完善制度规范和强化社会治理。",
        "锣鼓喧天，国乐齐鸣，歌舞翩翩……暖场舞蹈《乐舞汉唐》，用震撼的节奏、优美的舞姿，展现历史与时代交融、体育与文化共鸣，表达了咸阳人民最真挚的欢迎之情；",
        "3、参与投标的供应商可使用\"腾讯会议APP\"APP收看网上开标直播，法定代表人或授委托人可提前下载并通过其手机号进行注册，开标时登陆\"腾讯会议APP\"APP，交易中心将根据投标供应商提供的法定代表人或授委托人手机号邀请其观看开标实况。",
        "正所谓“一支独秀不是春，百花齐放春满院”，要想提高整体教学质量，唯有搞好集体备课。",
        "老师作为传道受业解惑者，自当为人师表，着力维护教育公平。",
        "7位青年干部广泛搜集党史资料，从中精挑细选感染人、打动人的党史故事，带领大家回顾党的百年历史，感知党的百年壮丽辉煌，用革命的故事打动人，用榜样的力量感染人，用今夕对比取信人。",
        "在家里，她能关注亲人的冷暖，以礼相让，让亲情其乐融融。",
        "近期，梁溪城管机动中队不断“烤”验"
    ]
    # m = Corrector(proper_name_path='')
    # for i in error_sentences:
    #     print(i, ' -> ', m.correct(i))
    # text = '预计：明天夜里到6号白天，多云到阴，部分地区有分散型阵雨或雷雨。'
    print('*' * 42)
    word_path = os.path.join(get_project_path(), 'knowledgebase/dict/custom_dict.txt')
    # proper_path=os.path.join(get_project_path(),'knowledgebase/dict/chengyu.txt')
    m = Corrector(word_freq_path=word_path,proper_name_path=word_path,min_proper_len=3)

    success,uncorrected,fail=0,0,0
    error_sentences = json.load(
        open(os.path.join(get_project_path(), 'models/model_MiduCTC/data/preliminary_a_data/preliminary_val.json'),
             encoding='utf-8'))

    success, uncorrected, fail = 0, 0, 0
    unchecked, total = 0, 0
    for ins in tqdm(error_sentences[:]):
        text=ins['source']
        total+=1
    # for text in tqdm(error_sentences[:]):
        # text='在舒适性方面，云南省对路面破损、平整度、车辙三项指标有一项或多项达不到“优”的22条共2940.866公里单幅高速公路进行了集中处治。'
        # text='激情高涨，深情并茂的用朗诵的方式表达自己对中华文化'
        corrected=m.correct(text,only_proper=True)
        if len(corrected[1])>0:
            if  corrected[0]==ins['target']:
                success+=1
            else:
                fail+=1
                print("Failed correct:",corrected)
        if corrected[0]==ins['source']:
            unchecked+=1
        # print(corrected)
    print("total,success,fail,unchecked:",total,success,fail,unchecked)
    # seg_list = jieba.cut(text, cut_all=False)  # 使用精确模式进行分词
    # print("/".join(seg_list))
