import torch
from tqdm import tqdm
import json
from models.model_MiduCTC.src import corrector, correctorV3
from tqdm import tqdm
import json
import numpy
from models.model_MiduCTC.src import corrector
# import mypycorrector
testa_data = json.load(open('../data/preliminary_a_data/preliminary_a_test_source.json',encoding='utf-8'))
# testa_data = json.load(open('../data/preliminary_a_data/preliminary_val.json',encoding='utf-8'))
# 模型
# correct= corrector.Corrector('../new_model/ctc_2022Y07M21D01H/epoch2,step62500,testf1_35_77%,devf1_35_77%')
# correct= corrector.Corrector('../new_model/ctc_2022Y07M21D08H/epoch4,step1,testf1_35_94%,devf1_35_94%')
# 词性融合
correct= corrector.Corrector('../model/ctc_2022Y07M27D23H/epoch1,step180,testf1_44_9%,devf1_44_9%')
submit = []
idx=0
equ_nums=0
s1,s2=0,0
exceed_max=0
print([1]+[2]*2)
for ins in tqdm(testa_data[:]):
    corrected_sent = correct(ins['source'])
    if len(ins['source'])>128:
        exceed_max+=1
        print(len(ins['source']))
    # corrected_sent2 = correct2(ins['source'])
    # print(corrected_sent2)
    # if corrected_sent[0]==corrected_sent2[0]:
    #     equ_nums+=1
    # else:
    #     if ins['type'] == 'negative':
    #         if ins['target']==corrected_sent[0]:
    #             s1+=1
    #         if ins['target']==corrected_sent2[0]:
    #             s2+=1
    submit.append({
        "inference": corrected_sent[0],
        "id": ins['id']
    })
    idx += 1
# print(submit,equ_nums,idx)
print("exceed,total nums:",exceed_max,idx)
print(s1,s2)
json.dump(submit, open('../data/preliminary_a_data/output/preliminary_a_test_source.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

