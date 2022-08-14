import os

from ProjectPath import get_project_path
from models.mypycorrector.corrector import Corrector

error_sentences = [
        "桃李天下！"
        "确保防控不松懈、疫情不反殚",
    "#醉汉连续3次跳楼累惨消防员#】援引天津消防消息：8月20日，天津北辰，一男子醉酒后一而再再三跑到高处跳“楼”，消防员刚归队又出警，轮番上阵，反复救援其三次。",
        "而该车辆荷载人数仅7人",
        "青岛市人民检查院指控，1999年至2020年，董宏先后担任海南省委副秘书长、北京市政府副秘书长、中央巡视组副组长等职务，为他人非法谋利，收受财物4.6亿余元。",
        "以城为镜，弘扬中华民主传统美德，让我们一起用实际行动为信阳的文明发展贡献出自己的一份力量！",
    "警方当下立断，让其父亲将梯子架在屋檐旁，并在梯子底下埋伏一个突击小组。",
    "2021年8月9日傍晚6时左右，虹桥镇万源新城3期117号门口的非机动车车棚里一辆电动自行车充电时突然自燃，车内的电瓶发生爆炸，火势一簇而上，又因当天风力强劲，火势迅速向西蔓延。",
    "自禄口机场暴发疫情以来，潘金海积极响应号召，招募志愿者，成立美年大健康志愿者服务小分队，组织防疫培训、帮忙搭建遮阳棚、协助核酸采集……每天忙得马不停歇。",
    "走进县直机关入党积极分子暨发展对象培训班，和瑞庭同志用自己的亲生经历为这92名入党积极分子上了一堂深刻的爱国主义教育课。"
    ]
proper_path=os.path.join(get_project_path(),'knowledgebase/dict/custom_dict.txt')
m = Corrector(custom_word_freq_path=proper_path,proper_name_path=proper_path,min_proper_len=4)
for i in error_sentences:
    print(i, ' -> ', m.correct(i,only_proper=True))