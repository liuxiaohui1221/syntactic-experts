import itertools
import os

import jieba
import pypinyin

from ProjectPath import get_project_path

user_dict = "knowledgebase/dict/custom_dict.txt"
user_dict=os.path.join(get_project_path(),user_dict)
# jieba.load_userdict(user_dict)

wss=["让我们一起用实际行动为信阳的文明发展贡献出自己的一份力量！",
    "警方当下立断，让其父亲将梯子架在屋檐旁，并在梯子底下埋伏一个突击小组。",
    "2021年8月9日傍晚6时左右，虹桥镇万源新城3期117号门口的非机动车车棚里一辆电动自行车充电时突然自燃，车内的电瓶发生爆炸，火势一簇而上，又因当天风力强劲，火势迅速向西蔓延。",
    "自禄口机场暴发疫情以来，潘金海积极响应号召，招募志愿者，成立美年大健康志愿者服务小分队，组织防疫培训、帮忙搭建遮阳棚、协助核酸采集……每天忙得马不停歇。",
    "走进县直机关入党积极分子暨发展对象培训班，和瑞庭同志用自己的亲生经历为这92名入党积极分子上了一堂深刻的爱国主义教育课。",
     "老师作为传道受业解惑者，自当为人师表，着力维护教育水平。",
     "发布了头条文章：《广东公安八大专项行动｜古玩骗局“换装上线”，警方提醒请勿利令智",
     "江苏省长吴政隆日前到扬州检查指导疫情防控工作，他说这次扬州疫情发生早、发现比较晚，在人员聚集的密闭场所，老年人居多，现在情况尚未见底。"
    ]
for ws in wss:
    seg_list = jieba.cut(ws, cut_all=False)  # 使用精确模式进行分词
    print("/".join(seg_list))

    print(jieba.lcut(ws))

    arr=[]
    for w in "国乐齐鸣":
        a1=pypinyin.pinyin(w, errors='ignore', heteronym=True, style=pypinyin.TONE3)
        arr.extend(a1)
    pysTone = list(itertools.product(*arr))
    print(arr,pysTone)

    print(pypinyin.pinyin("国乐齐鸣", errors='ignore', heteronym=True, style=pypinyin.TONE3))

pinyin_arr=pypinyin.pinyin("国乐齐鸣", style=pypinyin.Style.NORMAL, heteronym=True)
print(pinyin_arr)