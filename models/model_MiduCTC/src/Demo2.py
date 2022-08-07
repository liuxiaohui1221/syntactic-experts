import torch
from tqdm import tqdm
import json
from models.model_MiduCTC.src import corrector
import pycorrector
# testa_data = json.load(open('../data/preliminary_a_data/preliminary_a_test_source.json',encoding='utf-8'))
val_data = json.load(open('../data/preliminary_a_data/preliminary_val.json',encoding='utf-8'))
# val_data = json.load(open('../data/preliminary_a_data/preliminary_extend_train.json',encoding='utf-8'))

correct= corrector.Corrector('../model/ctc_2022Y07M27D23H/epoch1,step180,testf1_44_9%,devf1_44_9%')
print(correct("水灾和旱灾会破坏农业生产，造成粮食欠收。"))
submit = []
total=0
pred_right=0
# for ins in tqdm(val_data[:]):
#     if len(ins['source']) != len(ins['target']):
#         continue
#     total += 1
#
#     # tuple2 = nlp(ins['source'])
#     tuple2=correct(ins['source'])
#     if tuple2[0] == ins['target']:
#         pred_right += 1
#     submit.append({
#         "inference": tuple2[0],
#         "id": ins['id']
#     })
# print(pred_right,total,pred_right/total)
# for ins in tqdm(val_data[:]):
#     corrected_sent = correct(ins['source'])
#     corrected_sent2, detail = mypycorrector.correct(ins['source'])
#     if corrected_sent[0]==ins['source'] and  len(detail)==0:
#         continue
#     if corrected_sent[0]==corrected_sent2:
#         continue
#     print(ins['type']=='positive',corrected_sent[0]==ins['source'],corrected_sent,corrected_sent2,detail)
#     # 当判断正确则继续使用pycorrector检测
#     if corrected_sent[0] == ins['source']:
#         submit.append({
#             "inference": corrected_sent2,
#             "id": ins['id']
#         })
#     else:
#         submit.append({
#             "inference": corrected_sent[0],
#             "id": ins['id']
#         })
#     idx += 1
# json.dump(submit, open('../data/preliminary_a_data/output/preliminary_a_test_source.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)