# -*- coding: utf-8 -*-
"""
@Time   :   2021-02-03 21:57:15
@File   :   correct_demo.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import argparse
import json
import sys

from tqdm import tqdm

sys.path.append('../..')
from macbert_corrector import MacBertCorrector
from pycorrector import config


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--macbert_model_dir", default='output/macbert4csc',
                        type=str,
                        help="MacBert pre-trained model dir")
    args = parser.parse_args()

    m = MacBertCorrector(args.macbert_model_dir)
    # testa_data = json.load(open('./preliminary_a_test_source.json', encoding='utf-8'))
    testa_data = json.load(open('../model_MiduCTC/data/preliminary_a_data/preliminary_val.json', encoding='utf-8'))
    # testa_data = json.load(open('./preliminary_extend_train.json', encoding='utf-8'))
    submit=[]
    pred_right=0
    total=1
    for ins in tqdm(testa_data[:]):
        # if len(ins['source'])==len(ins['target']):
        #     continue
        # total+=1
        tuple2=m.macbert_correct_recall(ins['source'],val_target=ins['target'])
        # if tuple2[0]==ins['target']:
        #     pred_right+=1
        submit.append({
            "inference": tuple2[0],
            "id": ins['id']
        })
    #
    # json.dump(submit, open('./output/preliminary_a_test_source.json', 'w', encoding='utf-8'),
    #               ensure_ascii=False, indent=4)
    print(pred_right, total, pred_right/total)


    i = m.macbert_correct_recall('今新情很好')
    print(i[0])
    #
    # i = nlp('少先队员英该为老让座')
    # print(i)
    #
    # i = nlp('机器学习是人工智能领遇最能体现智能的一个分知。')
    # print(i)
    #
    # i = nlp('机其学习是人工智能领遇最能体现智能的一个分知。')
    # print(i)
    #
    # print(nlp('老是较书。'))
    # print(nlp('遇到一位很棒的奴生跟我聊天。'))


if __name__ == "__main__":
    main()
