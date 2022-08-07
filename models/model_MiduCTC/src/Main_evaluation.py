import numpy
import torch
from tqdm import tqdm
import json
from models.model_MiduCTC.src import corrector
from models.model_MiduCTC.src import correctorV3
# import mypycorrector
# val_data = json.load(open('../data/preliminary_a_data/preliminary_a_test_source.json',encoding='utf-8'))
val_data = json.load(open('../data/preliminary_a_data/preliminary_val.json',encoding='utf-8'))
# val_data = json.load(open('../data/preliminary_a_data/preliminary_train.json',encoding='utf-8'))

# correct= corrector.Corrector('../new_model/ctc_2022Y07M21D08H/epoch4,step1,testf1_35_94%,devf1_35_94%')
# correct= corrector.Corrector('../pretrained_model/chinese-roberta-wwm-ext')
# correct= corrector.Corrector('../new_model/ctc_2022Y07M22D23H/epoch2,step1,testf1_39_94%,devf1_39_94%')
correct= corrector.Corrector('../model/ctc_2022Y07M25D18H/epoch3,step1,testf1_46_7%,devf1_49_45%')
# correct= corrector.Corrector('../new_model/ctc_2022Y07M25D17H/epoch3,step1,testf1_47_02%,devf1_50_7%')
submit = []
acc_idx,idx=0,0
c_acc=numpy.array([0,0])
c_recall=numpy.array([0,0])
d_recall=numpy.array([0,0])
d_precision=numpy.array([0,0])
exceedLen,exceedLargeLen=0,0
for ins in tqdm(val_data[:]):
    if len(ins['source']) > 128 and len(ins['source']) < 200  :
        exceedLen+=1
    elif len(ins['source']) > 200:
        exceedLargeLen+=1
    corrected_sent = correct(ins['source'])
    if corrected_sent[0]==ins['target']:
        c_recall[0] += 1
    if corrected_sent[0]!=ins['source']:
        acc_idx += 1
    # print(corrected_sent)
    if ins['type'] == 'negative' and corrected_sent[0]!=ins['source']:
        c_acc[0] += 1
    if ins['type'] == 'negative' and corrected_sent[0] == ins['target']:
        d_precision[0] += 1
    idx += 1
print(exceedLen,exceedLargeLen)
print("CTC:pycor acc：",c_acc*1.0/acc_idx)
print("CTC:pycor c_recall：",c_recall*1.0/idx)
print("CTC:pycor d_precision：",d_precision*1.0/idx)
# json.dump(submit, open('../data/preliminary_a_data/output/preliminary_a_test_source.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)