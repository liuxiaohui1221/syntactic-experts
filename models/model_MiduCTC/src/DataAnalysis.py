import os

import torch
from tqdm import tqdm
import json
from tqdm import tqdm
import json
import pycorrector
# testa_data = json.load(open('../data/preliminary_a_data/preliminary_a_test_source.json',encoding='utf-8'))
# testa_data = json.load(open('../data/example_input.json',encoding='utf-8'))
from models.mypycorrector import corrector

testa_data = json.load(open('../data/preliminary_a_data/preliminary_val.json',encoding='utf-8'))
# 模型
# correct= corrector.Corrector('../new_model/ctc_2022Y07M21D01H/epoch2,step62500,testf1_35_77%,devf1_35_77%')
# correct= corrector.Corrector('../new_model/ctc_2022Y07M21D08H/epoch4,step1,testf1_35_94%,devf1_35_94%')
# 词性融合
# correct= corrector.Corrector('../new_model/ctc_2022Y07M22D04H/epoch2,step1,testf1_33_97%,devf1_33_97%')
# correct= corrector.Corrector('../model/ctc_2022Y07M22D23H/epoch2,step1,testf1_39_94%,devf1_39_94%')
correct= corrector.Corrector('../model/ctc_2022Y07M27D23H/epoch2,step63,testf1_44_99%,devf1_44_99%')

def train_analysis():
    testa_data = json.load(open('../data/preliminary_a_data/preliminary_train.json',encoding='utf-8'))
    pos,neg=0,0
    # 统计正负样本比例
    for ins in tqdm(testa_data[:]):
        if ins['source']==ins['target']:
            pos+=1
        else:
            neg+=1
    print(pos,neg)
train_analysis()

### 训练集或验证集中的假负例筛选：1.source和target相等但为negative的，或不等为positive的（继续分析可能为两正例？）--->手动修改并重新训练
def check_data_hard_neg():
    total_hard_neg = 0
    for ins in tqdm(testa_data[:]):
        if ins['source'] == ins['target'] and ins['type'] == 'negative':
            total_hard_neg += 1
            print(ins['id'])
        if ins['source'] != ins['target'] and ins['type'] == 'positive':
            total_hard_neg += 1
            print(ins['id'])
    print(total_hard_neg)

neg_right = []
neg_error = []
pos_right = []
pos_error = []

# 分析CTC模型预测能力：预测错误与正确的数据集分布
def show_CTC_predict_error_data(type):
    neg_predict_pos,neg_predict_other,idx = 0, 0, 0
    neg_predict_pos_missing,neg_predict_pos_replaceEqlLen,neg_predict_pos_redundancing=0,0,0
    neg_predict_other_missing, neg_predict_other_replaceEqlLen, neg_predict_other_redundancing = 0, 0, 0
    pos_predict_neg,pos_nums,neg_nums=0,0,0
    for ins in tqdm(testa_data[:]):
        if type=="CTC":
            # CTC纠错
            corrected_sent = correct(ins['source'])[0]
        else:
            # pycorrector纠错
            corrected_sent, detail = pycorrector.correct(ins['source'])

        if ins['type']=='negative':
            neg_nums+=1
            if corrected_sent == ins['target']:
                # 预测正确
                neg_right.append({"source": ins['source'],"inference": corrected_sent,"id": ins['id']})
            else:
                # 预测为正例或纠错不对
                # 统计缺字补全问题，以及同音替换问题分布
                if corrected_sent == ins['source']:
                    neg_predict_pos+=1
                    if len(ins['source'])<len(ins['target']): #missing
                        neg_predict_pos_missing+=1
                    elif len(ins['source'])==len(ins['target']): #replace equal len 同音替换问题
                        neg_predict_pos_replaceEqlLen+=1
                    else:
                        neg_predict_pos_redundancing+=1
                else:
                    neg_predict_other+=1
                    if len(ins['source'])<len(ins['target']): #missing
                        neg_predict_other_missing+=1
                    elif len(ins['source'])==len(ins['target']): #replace equal len 同音替换问题
                        neg_predict_other_replaceEqlLen+=1
                    else:
                        neg_predict_other_redundancing+=1
                neg_error.append({
                    "source": ins['source'],
                    "target": ins['target'],
                    "infere": "无错" if corrected_sent == ins['source'] else corrected_sent,
                    "id": ins['id']
                })
        else:
            pos_nums+=1
            if corrected_sent == ins['target']:
                # 预测正确
                pos_right.append({
                    "infere": corrected_sent,
                    "id": ins['id']
                })
            else:
                # 预测错误：无错判有错
                pos_predict_neg+=1
                pos_error.append({
                    "source": ins['source'],
                    "target": ins['target'],
                    "infere": "无错" if corrected_sent == ins['source'] else corrected_sent,
                    "id": ins['id']
                })
        idx += 1
    print("有错误样本预测统计：")
    print("[有错判无错,有错纠不准,有错数,idx]: %s,%s,%s,%s " %(neg_predict_pos, neg_predict_other, neg_nums,idx))
    print("[有错判无错,有错纠不准]比例: %.2f,%.2f " % (1.0*neg_predict_pos/idx, 1.0*neg_predict_other/idx))
    print("有错误样本中，判为无错问题：")
    print("[缺失问题,同等替换问题,冗余问题,总数]: %.2f,%.2f,%.2f,%.2f "
          % (neg_predict_pos_missing, neg_predict_pos_replaceEqlLen, neg_predict_pos_redundancing, neg_predict_pos))

    if neg_predict_pos>0:
        print(
            "[缺失问题,同等替换问题,冗余问题,总数]比例: %.2f,%.2f,%.2f "
            % (1.0*neg_predict_pos_missing/neg_predict_pos, 1.0*neg_predict_pos_replaceEqlLen/neg_predict_pos, 1.0*neg_predict_pos_redundancing/neg_predict_pos))
    print("有错误样本中，纠错不对问题：")
    print(
        "[缺失问题,同等替换问题,冗余问题,总数]: %s,%s,%s,%s "
        % (neg_predict_other_missing, neg_predict_other_replaceEqlLen, neg_predict_other_redundancing, neg_predict_other))
    if neg_predict_other > 0:
        print(
            "[缺失问题,同等替换问题,冗余问题,总数]比例: %.2f,%.2f,%.2f "
            % (1.0 * neg_predict_other_missing / neg_predict_other, 1.0 * neg_predict_other_replaceEqlLen / neg_predict_other,
               1.0 * neg_predict_other_redundancing / neg_predict_other))

    print("[无错判有错,无错数，比例]：%s,%s,%.2f"%(pos_predict_neg,pos_nums,pos_predict_neg/pos_nums))
    # print(s1, s2)
    json.dump(neg_right, open('../data/preliminary_a_data/output/preliminary_extend_neg_right.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    json.dump(neg_error, open('../data/preliminary_a_data/output/preliminary_extend_neg_error.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
    json.dump(pos_right, open('../data/preliminary_a_data/output/preliminary_extend_pos_right.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
    json.dump(pos_error, open('../data/preliminary_a_data/output/preliminary_extend_pos_error.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
show_CTC_predict_error_data("CTC")
print("-"*20)
# show_CTC_predict_error_data("PYC")
