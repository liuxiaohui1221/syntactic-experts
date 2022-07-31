import json
import os
from difflib import SequenceMatcher

from tqdm import tqdm
from transformers import BertTokenizer
from ProjectPath import get_project_path


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
    tokenizer = BertTokenizer.from_pretrained('../model/macbert/pretrained/macbert4csc')
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
        results.append({
            "id":"-",
            "original_text":ins['source'][:158],
            "wrong_ids":wrongIds,
            "correct_text":ins['target'][:158]
        })
        # if len(results) % 50000==0:
        #     n+=1
        #     json.dump(results, open(os.path.join(get_project_path(), outPath+str(n)+'.json'), 'w', encoding='utf-8'),
        #               ensure_ascii=False, indent=4)
        #     results=[]
    json.dump(results, open(os.path.join(get_project_path(), outPath), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
    print(len(results))
transfer_to_macbert_format('model/model_MiduCTC/data/preliminary_a_data/preliminary_train_gen_words.json',
                           'model/macbert/output/preliminary_train_words_spell.json')
#
# transfer_to_macbert_format('model/model_MiduCTC/data/preliminary_a_data/preliminary_val.json',
#                            'model/macbert/output/preliminary_val_spell.json')

# transfer_to_macbert_format('model/model_MiduCTC/data/preliminary_a_data/preliminary_extend_train_gen_words.json',
#                            'model/macbert/output/preliminary_extend_train_spell.json')



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
