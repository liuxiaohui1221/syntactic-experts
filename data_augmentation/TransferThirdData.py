import json
import os
from difflib import SequenceMatcher

from tqdm import tqdm
from transformers import BertTokenizer
from ProjectPath import get_project_path
from models.macbert.util.common import getEdits


def transfer(inPath,outPath):
    texts=[]
    with open(os.path.join(get_project_path(), inPath),'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            json_line=json.loads(line)
            # print(json_line['ID'])
            texts.append({
                "id":int(json_line['ID'][2:]),
                "source":json_line['source'],
                "target":json_line['target'],
                "type":"negative"
            })
    json.dump(texts, open(os.path.join(get_project_path(), outPath), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
# transfer('data_augmentation/third/train_large_v2.json','model/model_MiduCTC/data/preliminary_a_data/CTC2021_train.json')

def transfer_to_macbert_format(inPath,outPath):
    inputData = json.load(open(os.path.join(get_project_path(),inPath),'r', encoding='utf-8'))
    results=[]
    n=0
    tokenizer = BertTokenizer.from_pretrained('../models/macbert/pretrained/macbert4csc')
    for ins in tqdm(inputData[:]):
        if len(ins['source'])!=len(ins['target']):
            continue

        src_embeddingids=tokenizer(ins['source'], padding=True, return_tensors='pt')['input_ids']
        trg_embeddingids = tokenizer(ins['target'], padding=True, return_tensors='pt')['input_ids']
        if src_embeddingids.shape[1]!=trg_embeddingids.shape[1]:
            print('diff:',ins['source'],ins['target'])
            continue
        r = SequenceMatcher(None, ins['source'], ins['target'])
        diffs = r.get_opcodes()
        wrongIds=[]

        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if tag=='replace':
                wrongIds.extend(range(i1,i2))
        if len(wrongIds)==0:
            continue
        results.append({
            "id":"-",
            "original_text":ins['source'][:200],
            "wrong_ids":wrongIds,
            "correct_text":ins['target'][:200]
        })
        # if len(results) % 50000==0:
        #     n+=1
        #     json.dump(results, open(os.path.join(get_project_path(), outPath+str(n)+'.json'), 'w', encoding='utf-8'),
        #               ensure_ascii=False, indent=4)
        #     results=[]
    json.dump(results, open(os.path.join(get_project_path(), outPath), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
    print(len(results))
# transfer_to_macbert_format('models/ECSpell/Data/traintest/final_train.json',
#                            'models/macbert/output/final_train_spell.json')
transfer_to_macbert_format('models/ECSpell/Data/traintest/final_val.json',
                           'models/macbert/output/final_val_spell.json')
#
# transfer_to_macbert_format('model/model_MiduCTC/data/preliminary_a_data/preliminary_val.json',
#                            'model/macbert/output/preliminary_val_spell.json')

# transfer_to_macbert_format('model/model_MiduCTC/data/preliminary_a_data/preliminary_extend_train_gen_words.json',
#                            'model/macbert/output/preliminary_extend_train_spell.json')

def transfer_to_ecsspell_format(inPath,outPath):
    inputData = json.load(open(os.path.join(get_project_path(), inPath), 'r', encoding='utf-8'))
    results = []
    n = 0
    tokenizer = BertTokenizer.from_pretrained('../models/macbert/pretrained/macbert4csc')
    for ins in tqdm(inputData[:]):
        if len(ins['source']) != len(ins['target']):
            continue

        src_embeddingids = tokenizer(ins['source'], padding=True, return_tensors='pt')['input_ids']
        trg_embeddingids = tokenizer(ins['target'], padding=True, return_tensors='pt')['input_ids']
        if src_embeddingids.shape[1] != trg_embeddingids.shape[1]:
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
        line=[str(type),ins['source'],ins['target']]
        results.append("\t".join(line))
    with open(os.path.join(get_project_path(), outPath),'w',encoding='utf-8') as f:
        for row in results:
            f.write(row+'\n')

# transfer_to_ecsspell_format('model/model_MiduCTC/data/preliminary_a_data/preliminary_val.json',
#                            'data_augmentation/preliminary_val_ecspell.test')
# transfer_to_ecsspell_format('models/model_MiduCTC/data/preliminary_a_data/preliminary_extend_train.json',
#                            'data_augmentation/preliminary_extend_train_ecspell.test')
# transfer_to_ecsspell_format('model/model_MiduCTC/data/preliminary_a_data/preliminary_train.json',
#                            'data_augmentation/preliminary_train_ecspell.train')

def transfer_from_macbert_format(inPath,outPath):
    texts = []
    dicts=json.load(open(os.path.join(get_project_path(), inPath),encoding='utf-8'))
    for index,line in enumerate(dicts[:]):
        # print(json_line['ID'])
        texts.append({
            "id": index,
            "source": line['original_text'],
            "target": line['correct_text'],
            "type": "negative"
        })
    json.dump(texts, open(os.path.join(get_project_path(), outPath), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
    print(len(texts))
# transfer_from_macbert_format('data_augmentation/third/csc_sample/train.json','model/model_MiduCTC/data/preliminary_a_data/csc-train.json')
# transfer_from_macbert_format('data_augmentation/third/csc_sample/dev.json','models/model_MiduCTC/data/preliminary_a_data/csc-dev.json')

def filter_loss_from_val():
    # 从验证集和extend_train中过滤出缺字错误及一半的：replace错误、无错样本
    inpaths=[
        'models/model_MiduCTC/data/preliminary_a_data/final_val.json',
        'models/model_MiduCTC/data/preliminary_a_data/preliminary_val.json',
             'models/model_MiduCTC/data/preliminary_a_data/preliminary_extend_train.json']
    filterd_data=[]
    for inpath in inpaths:
        dicts = json.load(open(os.path.join(get_project_path(), inpath), encoding='utf-8'))
        for row in dicts:
            edits=getEdits(row['source'],row['target'])
            for edit in edits:
                if edit[0]=='insert':
                    filterd_data.append(row)
                    break
    take_num=0
    take_loss=len(filterd_data)
    for inpath in inpaths:
        dicts = json.load(open(os.path.join(get_project_path(), inpath), encoding='utf-8'))
        for row in dicts:
            if take_num>take_loss*2:
                break
            edits = getEdits(row['source'], row['target'])
            for edit in edits:
                if edit[0]=='insert' or edit[0]=='delete':
                    continue
                # replace负例当做正例
                row['target']=row['source']
                filterd_data.append(row)
                take_num+=1
                break
    print("Got size",len(filterd_data))
    outpath=os.path.join(get_project_path(), 'models/model_MiduCTC/data/preliminary_a_data/only_loss_val.json')
    json.dump(filterd_data, open(outpath, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
filter_loss_from_val()
