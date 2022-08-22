import csv
import os.path
from difflib import SequenceMatcher

import pycorrector
import torch
from ltp import LTP
from tqdm import tqdm
import json

from ProjectPath import get_project_path
from knowledgebase.tencent.SentenceSimilarity import WordSentenceSimliarity
from models.ECSpell.Code.ProjectPath import get_ecspell_path
from models.macbert.macbert_corrector import MacBertCorrector
from models.macbert.util.common import removeDuplicate, fenciCorrect
from tqdm import tqdm
import json
import numpy
from models.model_MiduCTC.src import corrector
import argparse

from models.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf
from models.mypycorrector import config
from models.mypycorrector.corrector import Corrector

def predictAgainM1M2PyDict(ctc1_text, mac2_text, ins, pydict_text,fenci):
    if pydict_text!=ins['source']:
        # 再次去除重复词:由于缺字匹配可能导致的多字问题
        pydict_text = removeDuplicate(fenci, pydict_text)
        return pydict_text
    if len(ctc1_text)!=len(ins['source']):
        return ctc1_text
    elif len(ins['ecspell'])!=len(ins['source']):
        # 词向量检测
        return ctc1_text
    else:
        # 拼写检测问题
        if mac2_text!=ins['ecspell'] and ins['ecspell']==ins['source']:
            # ecspell模型漏检
            return mac2_text
    return ins['ecspell']

# if __name__ == "__main__":
def eval2(midu_ctc_model_dir='models/model_MiduCTC/pretrained_model/epoch3,step1,testf1_61_91%,devf1_55_17%',
          macbert_model_dir='models/macbert/macbert4csc'):
    parser = argparse.ArgumentParser()
    macbert_model_dir=os.path.join(get_project_path(),macbert_model_dir)
    parser.add_argument("--macbert_model_dir", default=macbert_model_dir,
                        type=str,
                        help="MacBert pre-trained model dir")
    args = parser.parse_args()
    fenci=VocabConf().jieba_singleton
    ctc_correct = corrector.Corrector(
        os.path.join(get_project_path(),midu_ctc_model_dir)
        , ctc_label_vocab_dir=os.path.join(get_project_path(), 'models/model_MiduCTC/src/baseline/ctc_vocab'))
    m = MacBertCorrector(args.macbert_model_dir)
    # proper_path = os.path.join(get_project_path(), 'knowledgebase/dict/chengyu.txt')
    confusion_path = os.path.join(get_project_path(), 'models/mypycorrector/data/confusion_pair.txt')
    word_path = os.path.join(get_project_path(), 'knowledgebase/dict/custom_dict.txt')
    m4 = Corrector(custom_confusion_path=confusion_path, word_freq_path=word_path, proper_name_path=word_path)
    testa_data = json.load(open(os.path.join(get_ecspell_path(),'Code/Results/ecspell/results/checkpoint-final_test_source.json'),encoding='utf-8'))
    submit_2=[]
    wss = WordSentenceSimliarity()
    # ltp分词器
    ltp = LTP(pretrained_model_name_or_path=config.ltp_model_path)
    for ins in tqdm(testa_data[:]):
        # 去除重复词
        src_text = removeDuplicate(fenci, ins['source'])
        if src_text==ins['source']:
            # 比较拼写纠错问题
            corrected_sent = ctc_correct([src_text])
            loss_corrected_send=[""]
            corrected_sent3 = m.macbert_correct_recall(src_text,val_target=ins.get('target',None),topk=20)

            corrected_sent4, detail = m4.correct(src_text)
        else:
            # 再次去除重复词
            src_text = removeDuplicate(fenci, src_text)
            corrected_sent4=src_text
            corrected_sent=src_text
            loss_corrected_send=src_text
            corrected_sent2=src_text
            corrected_sent3=src_text
            final_corrected=src_text
            detail=""

        final_corrected3 = predictAgainM1M2PyDict(corrected_sent[0],corrected_sent3[0],ins,corrected_sent4,fenci)
        # 后处理
        # 1.前后分词对比
        final_corrected3 = fenciCorrect(ltp, ins['source'], final_corrected3)

        submit_2.append({
            "inference": final_corrected3,
            "id": ins['id']
        })

    json.dump(submit_2, open(os.path.join(get_project_path(),'models/final_test_inference.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)



