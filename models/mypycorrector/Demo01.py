import json
import pandas as pd
import numpy
import torch
from tqdm import tqdm
from models.model_MiduCTC.src import thulac, corrector
from models.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf
from models.model_MiduCTC.src.baseline.tokenizer import CtcTokenizer
import pycorrector
# train_data = json.load(open('./preliminary_a_data/preliminary_train.json',encoding='utf-8'))
# exttrain_data = json.load(open('./preliminary_a_data/preliminary_extend_train.json',encoding='utf-8'))
testa_data = json.load(open('../model_MiduCTC/data/preliminary_a_data/preliminary_val.json',encoding='utf-8'))
# testa_data = json.load(open('./preliminary_a_data/preliminary_a_test_source.json',encoding='utf-8'))

vocabdir='../model_MiduCTC/src/baseline/ctc_vocab'
correct_old= corrector.Corrector('../model_MiduCTC/model/ctc_2022Y07M21D08H/epoch4,step1,testf1_35_94%,devf1_35_94%',ctc_label_vocab_dir=vocabdir)
correct = corrector.Corrector('../model_MiduCTC/model/ctc_2022Y07M22D23H/epoch2,step1,testf1_39_94%,devf1_39_94%',ctc_label_vocab_dir=vocabdir)

# 对比两个训练模型预测差异
total,diff,v2_right,v1_right=0,0,0,0
new_corrected=[]
old_corrected=[]
double_errored=[]
double_error=0
for ins in tqdm(testa_data[:]):
    corrected_send_old = correct_old(ins['source'])[0]
    corrected_send = correct(ins['source'])[0]
    if corrected_send_old!=corrected_send:
        if corrected_send==ins['target']:
            new_corrected.append({
                "source": ins['source'],
                "target": corrected_send,
                "type": ins['type']
            })
            v2_right+=1
        elif corrected_send_old==ins['target']:
            old_corrected.append({
                "source": ins['source'],
                "target": corrected_send_old,
                "type": ins['type']
            })
            v1_right+=1
        diff+=1
        print("source,target,new_predict,old_predict",ins['source'],corrected_send==ins['target'],corrected_send_old==ins['target'])
    else:
        if corrected_send!=ins['target']:
            double_error+=1
            double_errored.append({
                "source": ins['source'],
                "target": ins['target'],
                "refere":corrected_send,
                "type": ins['type']
            })
    total+=1
print("total,diff,v2_right,v1_right,double_error:",total,diff,v2_right,v1_right,double_error)
json.dump(new_corrected, open('../preliminary_val_v2.json', 'w',
                                      encoding='utf-8'),
                      ensure_ascii=False, indent=4)
json.dump(old_corrected, open('../preliminary_val_v1.json', 'w',
                                          encoding='utf-8'),
                      ensure_ascii=False, indent=4)
json.dump(double_errored, open('../preliminary_val_doubleErr.json', 'w',
                                          encoding='utf-8'),
                      ensure_ascii=False, indent=4)
# 乱序
# text='就连我怀孕时的候，不顾医生的一再叮嘱，还是忍不住要跑去吃铜锅，吃涮羊肉。'
# 缺字
text='认罪伏法定边法院经审理认为，被告人杨某暴力袭击正在依法执行职务的人民警察，其行为已构成袭警罪。'
corrected_sent, detail = pycorrector.correct(text)
corrected2=correct(corrected_sent)[0]

print(corrected_sent, detail)
print("CTC:",corrected2,corrected_sent==corrected2)
# 现在上学无非是之后能有个好的机会拿到称心的工作赚到钱过的好。 [('咯', '个', 11, 12)]

# submit = []
# idx=0
# success=0
# c_recall=0
# for ins in tqdm(testa_data[:]):
#     corrected_sent, detail = mypycorrector.correct(ins['source'])
#     # submit.append({
#     #     "inference": corrected_sent,
#     #     "id": ins['id']
#     # })
#     # corrected_sent2, detail = mypycorrector.correct(ins['source'])
#     if corrected_sent == ins['target']:
#         success += 1
#     print(corrected_sent[0])
#     if ins['type'] == 'negative' and len(detail)>0:
#         c_recall+=1
#     idx += 1
# print("CTC:pycor识别准确率：",success*1.0/idx)
# print("CTC:pycor recal：",c_recall*1.0/idx)
# json.dump(submit, open('./preliminary_a_data/output/preliminary_a_test_source.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)



vocab_types = ["n", "np", "ns", "ni", "nz", "m", "q", "mq", "t", "f", "s", "v", "a", "d", "h", "k", "i", "j", "r",
                   "c", "p", "u", "y"
        , "e", "o", "g", "w", "x"]
vocab_types = ["n", "np", "ns", "ni", "nz", "m", "q", "mq", "t", "f", "s", "v", "a", "d", "h", "k", "i", "j", "r",
                   "c", "p", "u", "y"
        , "e", "o", "g", "w", "x"]
vocab_type2id = {i: v for i, v in enumerate(vocab_types)}
vocab_id2type={v: i for i, v in enumerate(vocab_types)}
# print(vocab_type2id)
# print(vocab_id2type)

# thu1 = thulac.thulac()  # 默认模式
# def convert_word_to_property(token: str):
#     text = thu1.cut(token, text=True).split(sep=' ')  # 进行一句话分词
#     print(text, type(text))
#     w_propertys = numpy.zeros(128)
#     p_sum=0
#     for x_p in text:
#         xp = x_p.split(sep="_")
#         p_sum+=len(xp[0])
#         p2id = VocabConf.vocab_type2id[xp[len(xp) - 1]]
#         w_propertys[p_sum-1]=p2id
#     return w_propertys
# input_propertes=convert_word_to_property("我爱北京天安门")
# print(input_propertes)
tokenizer = CtcTokenizer.from_pretrained('../model_MiduCTC/model/ctc_2022Y07M21D08H/epoch4,step1,testf1_35_94%,devf1_35_94%')
vocab_types = ["unknow","n", "np", "ns", "ni", "nz", "m", "q", "mq", "t", "f", "s", "v", "a", "d", "h", "k", "i", "j",
                   "r",
                   "c", "p", "u", "y"
        , "e", "o", "g", "w", "x"]
vocab_id2type = {"[unused"+str(i+10)+"]": v for i, v in enumerate(vocab_types)}
vocab_type2id = {v: "[unused"+str(i+10)+"]" for i, v in enumerate(vocab_types)}
# print(vocab_id2type)
# print(vocab_type2id)
text = VocabConf.thulac_singleton.cut("我爱北京天安门", text=True).split(sep=' ')  # 进行一句话分词
w_propertys = []
for x_p in text:
    xp = x_p.split(sep="_")
    p2code = VocabConf.vocab_type2id.get(xp[len(xp) - 1], "unknow")
    w_propertys.append(tokenizer.vocab[p2code])
# print("".join(w_propertys))
# print(w_propertys)
# inputs = tokenizer("".join(w_propertys),64)
# print(inputs)
# print(inputs['input_ids'])
