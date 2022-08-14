import json
import os
from collections import defaultdict
from difflib import SequenceMatcher

from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer, AutoConfig

from models.ECSpell.Code import common_utils
from models.ECSpell.Code.ProjectPath import get_ecspell_path

def getTag2Id(load_pretrain_checkpoint):
    with open(os.path.join(load_pretrain_checkpoint, "labels.txt"), "r", encoding="utf-8") as f:
        labels = f.read().strip().split("\n")
    tag2id = {tag: id for id, tag in enumerate(labels)}
    return tag2id
#
# def isChinese(cchar):
#     if u'\u4e00' <= cchar <= u'\u9fff':
#         return True
#     else:
#        return False
def is_zh_punctuation(w):
    strip_chars = '？。，；：、'
    if w in strip_chars:
        return True
    return False
def is_number(w):
    if '0'<=w<='9':
        return True
    return False

def is_en(w):
    if 'a'<=w<='z' or 'A'<=w<='Z':
        return True
    return False


def isNotLabel(text,labels):
    for word in text:
        if common_utils.is_chinese_char(ord(word)) and word not in labels:
            return True
    return False


def transfer_to_ecsspell_format(inPath,outPath):
    inputData = json.load(open(os.path.join(get_ecspell_path(), inPath), 'r', encoding='utf-8'))
    results = []
    n = 0
    labels=[]
    with open(os.path.join(get_ecspell_path(), "Results/labels.txt"), "r", encoding="utf-8") as f:
        labels = f.read().strip().split("\n")
    model_name = os.path.join(get_ecspell_path(),'Transformers/glyce')
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
    for index,ins in enumerate(tqdm(inputData[:])):
        if len(ins['source']) != len(ins['target']):
            continue
        # 去除特殊符号及英文字母
        temp_source=''
        for word in ins['source']:
            if common_utils.is_chinese_char(ord(word)) == False and is_zh_punctuation(word) == False and is_number(word) == False :
                # print(word)
                continue
            temp_source+=word

        temp_target=''
        for word in ins['target']:
            if common_utils.is_chinese_char(ord(word))==False and is_zh_punctuation(word)==False and is_number(word)==False:
                # print(word)
                continue
            temp_target+=word
        if isNotLabel(ins['source'],labels):
            print("Filter contains unknow chinese:",ins['source'])
            continue
        if isNotLabel(ins['target'],labels):
            print("Filter contains unknow chinese:",ins['source'])
            continue
        ins['source']=temp_source
        ins['target']=temp_target
        src_embeddingids = tokenizer(list(ins['source']), is_split_into_words=True, return_offsets_mapping=True)['input_ids']
        trg_embeddingids = tokenizer(list(ins['target']), is_split_into_words=True, return_offsets_mapping=True)['input_ids']
        if len(src_embeddingids) != len(trg_embeddingids):
            print('diff:', ins['source'], ins['target'])
            continue
        r = SequenceMatcher(None, ins['source'], ins['target'])
        diffs = r.get_opcodes()
        wrongIds = []

        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if tag == 'replace':
                wrongIds.extend(range(i1, i2))
        type=0
        if len(wrongIds) == 0:
            type=0
        else:
            type=1
        # line=[str(type),ins['source'],ins['target']]
        results.append({
            "id": index,
            "source": "".join(ins['source']),
            "target": "".join(ins['target']),
            "type": "negative"
        })
        # if len(results)%50000==0:
        #     json.dump(results, open(os.path.join(get_ecspell_path(), 'Data/traintest/preliminary_train_gen_ecspell.train'+str(index+1)), 'w', encoding='utf-8'),
        #               ensure_ascii=False, indent=4)
        #     results=[]
    json.dump(results, open(os.path.join(get_ecspell_path(), outPath), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)

def together_file(inPath='Data/traintest',outPath='Data/traintest/preliminary_train_ecspell.json'):
    dictPaths = []
    inPath=os.path.join(get_ecspell_path(),inPath)
    for fn in os.listdir(inPath):
        if fn[-3:] == '000':
            print(fn)
            dictPaths.append(os.path.join(inPath, fn))
        if fn[-6:] == '400000':
            break
    all_data=[]
    for filepath in dictPaths:
        json_data = json.load(open(os.path.join(filepath), encoding='utf-8'))
        all_data.extend(json_data)
    print("Together rows:",len(all_data))
    json.dump(all_data, open(os.path.join(get_ecspell_path(),outPath), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
# together_file()
# transfer_to_ecsspell_format('Data/traintest/preliminary_val.json',
#                            'Data/traintest/preliminary_val_ecspell.test')
# transfer_to_ecsspell_format('Data/traintest/preliminary_extend_train.json',
#                            'Data/traintest/preliminary_extend_train_ecspell.test')
# transfer_to_ecsspell_format('Data/traintest/preliminary_train.json',
#                            'Data/traintest/preliminary_train_ecspell.train')

# transfer_to_ecsspell_format('Data/traintest/preliminary_train_gen_ecspell.train200000',
#                            'Data/traintest/preliminary_train_gen_ecspell.train200000v2')

# transfer_to_ecsspell_format('Data/traintest/csc-dev.json','Data/traintest/csc-dev_ecspell.json')
transfer_to_ecsspell_format('Data/traintest/csc-test.json','Data/traintest/csc-test_ecspell.json')