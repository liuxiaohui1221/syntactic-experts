import json
import os
from difflib import SequenceMatcher

from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer, AutoConfig
from models.ECSpell.Code.ProjectPath import get_ecspell_path

def getTag2Id(load_pretrain_checkpoint):
    with open(os.path.join(load_pretrain_checkpoint, "labels.txt"), "r", encoding="utf-8") as f:
        labels = f.read().strip().split("\n")
    tag2id = {tag: id for id, tag in enumerate(labels)}
    return tag2id
def transfer_to_ecsspell_format(inPath,outPath):
    inputData = json.load(open(os.path.join(get_ecspell_path(), inPath), 'r', encoding='utf-8'))
    results = []
    n = 0
    model_name = os.path.join(get_ecspell_path(),'Transformers/glyce')
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
    for ins in tqdm(inputData[:]):
        if len(ins['source']) != len(ins['target']):
            continue

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
        line=[str(type),ins['source'],ins['target']]
        results.append("\t".join(line))
    with open(os.path.join(get_ecspell_path(), outPath),'w',encoding='utf-8') as f:
        for row in results:
            f.write(row+'\n')

transfer_to_ecsspell_format('Data/traintest/preliminary_val.json',
                           'Data/traintest/preliminary_val_ecsspell.test')
transfer_to_ecsspell_format('Data/traintest/preliminary_extend_train.json',
                           'Data/traintest/preliminary_extend_train_ecsspell.test')
transfer_to_ecsspell_format('Data/traintest/preliminary_train.json',
                           'Data/traintest/preliminary_train_ecsspell.train')
