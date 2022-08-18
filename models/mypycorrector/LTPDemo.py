from ltp import LTP

ltp = LTP(pretrained_model_name_or_path="LTP/base2")  # 默认加载 Small 模型
# ltp.add_word()
# ltp = LTP(pretrained_model_name_or_path="LTP/small")
# 另外也可以接受一些已注册可自动下载的模型名(https://huggingface.co/LTP):
# 使用字典结果
# "多个地点及公共地方加强消毒　供澳鲜活食品数量正常#澳门特区政府市政署表示，该署已因应最新疫情采取以下措施：市政署已对确诊病例的居所及周边街道、大厦公共部门，以及工作场所周边街道进行了重点清洁消毒。"
text1="遍布在世界各地的海外华文媒体的读者群不能小觑"
text2="市政署已对确诊病例的居所及周边街道、大厦公共部份，以及工作场所周边街道进行了重点清洁消毒"
ws=[
    "延申阅读记者：张虞李鹏志编辑：胡梦莹编审：赵宗杰终审：李恩广总监制：肇慧茹",
    "一只航母造价130亿，美国背负27万亿外债，为何还能养得起",
    "田铺大塆映入眼帘",
    "如今，田铺大塆深耕乡村旅游，发展农村经济，走上致富的快车道。",
    "如今的田铺大塆，创客工坊林立，创意的火花在传统村落中碰撞。",
    "如今，乡村旅游已成为田铺大塆的主要支柱产业。",
    "青少年是否有必要接种接种新全病毒疫苗？",
    "世界都会将目光投向东方",
    "世界都市",
    "三大作风",
    text1,text2,
    "就是以联合国宪章宗旨和原则为基础的国际关系基本准则",
    "还代款",
    "还贷款",
    "立足法律监督智能",
    "集中处治",
    "以咬定青山不放松的执着、行百里者半九十的清醒不懈奋斗，敢于斗争、善于斗争，逢山开道、遇水架桥，中国人民孜孜以求的美好梦想终将成为现实。",
    "江苏省长吴政隆日前到扬州检查指导疫情防控工作，他说这次扬州疫情发生早、发现比较晚，在人员聚集的密闭场所，老年人居多，现在情况尚未见底。"
]
for text in ws:
    output = ltp.pipeline(text, tasks=["cws", "pos", "ner", "srl", "dep", "sdp"])
    print(output.cws)
    print(output.pos)
# print(output.sdp)
print("*"*50)
# print(output.srl)
# print(output.ner)

text2="市政署已对确诊病例的居所及周边街道、大厦公共部门，以及工作场所周边街道进行了重点清洁消毒"
output = ltp.pipeline(text2, tasks=["cws", "pos", "ner", "srl", "dep", "sdp"])
print(output.cws)
print(output.pos)

# 传统算法，比较快，但是精度略低
# ltp = LTP("LTP/legacy")
# cws, pos, ner = ltp.pipeline(
#     ["他叫汤姆去拿外衣。"], tasks=["cws", "pos", "ner"]
# ).to_tuple()
# print(cws, pos, ner)