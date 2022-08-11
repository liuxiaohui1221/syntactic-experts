import os
from transformers import AutoTokenizer
from tqdm import tqdm
from common_utils import clean_text
from models.ECSpell.Code.ProjectPath import get_ecspell_path


def data_to_token_classification(filenames, tokenizer, save_filename, reverse=False, overwrite=True):
    print(f"save file: {save_filename}")
    dir_name = os.path.dirname(save_filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    mode = "w" if overwrite else "a"
    f_save = open(save_filename, mode, encoding='utf-8')
    total_count = 0
    filter = 0
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for line in tqdm(f):
                items = line.split('\t')
                if len(items) != 3 or len(items[1]) == 0:
                    continue
                items = [x.strip() for x in items]
                if len(items[1]) != len(items[2]):
                    print(
                        'data mismatch, ignore! {}->{}'.format(items[1], items[2]))
                total_count += 1
                if not reverse:
                    pair = items[1:]
                else:
                    pair = items[:0:-1]
                tokens, labels = get_token_labels(tokenizer, clean_text(pair[0]), clean_text(pair[1]))
                temp = tokenizer(tokens, is_split_into_words=True, add_special_tokens=False)
                # 过滤掉会二次subword情况
                if len(temp["input_ids"]) != len(tokens):
                    filter += 1
                    continue
                for token, label in zip(tokens, labels):
                    f_save.write('{}\t{}\n'.format(token, label))
                f_save.write('\n')
    f_save.close()
    print(f"filter num = {filter}")
    return total_count


def get_token_labels(tokenizer, input_sent, output_sent, max_length=128):
    tokenize_result = tokenizer(
        input_sent, return_offsets_mapping=True, truncation=True, max_length=max_length)
    tokens = []
    labels = []
    for offsets in tokenize_result['offset_mapping']:
        if offsets[1] <= offsets[0]:
            continue
        input_token = input_sent[offsets[0]:offsets[1]]
        output_token = output_sent[offsets[0]:offsets[1]]
        labels.append(output_token)
        tokens.append(input_token)
    return tokens, labels


filemaps = {
    "sim": [
        os.path.join(get_ecspell_path(),"Data/traintest/preliminary_train.json"),
        # "csc_evaluation/builds/sim/nlg/preliminary_extend_train_ecsspell.txt", "csc_evaluation/builds/sim/nlg/preliminary_val_ecsspell.txt",
        # "csc_evaluation/data/basedata/simplified/nlg.txt"
        # "csc_evaluation/data/basedata/simplified/train2015.txt",
        # "csc_evaluation/data/basedata/simplified/train2014.txt",
        # "csc_evaluation/data/basedata/simplified/train2013.txt"
    ],
}
reverse = False
model_list = ["glyce"]

for model_name in model_list:
    tokenizer = AutoTokenizer.from_pretrained(f"../Transformers/{model_name}")
    for font_type, filenames in filemaps.items():
        save_dir = f'Data/traintest/{font_type}/{model_name}'
        print(f"Model name: {model_name}\tFont type: {font_type}")
        for filename in filenames:
            corpus_type = filename.split("/")[-2]
            print(f'Handle file: {filename}')
            total_count = data_to_token_classification(
                [filename], tokenizer, os.path.join(save_dir, corpus_type, os.path.basename(filename)), reverse=reverse)
            # total_count = data_to_token_classification(
            #     [filename], tokenizer, os.path.join(save_dir, os.path.basename(filename)), reverse=reverse)
            print(f'total count: {total_count}')
