import json
import os
import torch
from transformers import AutoTokenizer, BertForTokenClassification
import sys
sys.path.append('../../..')
import models.ECSpell.Code.common_utils as common_utils
from models.ECSpell.Code.ProjectPath import get_ecspell_path
from models.ECSpell.Code.model import ECSpell
from models.ECSpell.Code.pipeline import tagger
from models.ECSpell.glyce.dataset_readers.bert_config import Config
from models.ECSpell.Code.data_processor import py_processor
from models.ECSpell.Code.processor import Processor
from models.ECSpell.csc_evaluation.evaluate_utils import compute_metrics, official_compute_metrics
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evaluate(pred_filename, need_tokenize=True):
    results = []
    # data = common_utils.read_table_file(pred_filename, output_indexes=[0, 1, 2])
    all_pairs = json.load(open(pred_filename, encoding='utf-8'))
    for ins in all_pairs:
        line_src, line_tgt, line_pred = ins['source'],ins.get("target"),ins["ecspell"]
        if need_tokenize:
            line_src = line_src.split()
            line_tgt = line_tgt.split()
            line_pred = line_pred.split()

        line_src = list(line_src)
        if line_tgt:
            line_tgt = list(line_tgt)
        else:
            line_tgt = []
        line_pred = list(line_pred)

        results.append((line_src, line_tgt, line_pred))

    compute_metrics(results)
    official_compute_metrics(results)
    return

def _preprocess(src_filenames):
    """ preprocess data"""
    all_pairs = json.load(open(src_filenames[0],encoding='utf-8'))
    src_sents = [x['source'] for x in all_pairs]
    # record the indexes of not chinese characters
    vocab = []
    for sent in src_sents:
        line = [0 for _ in range(len(sent))]
        for i, c in enumerate(sent):
            if not common_utils.is_chinese_char(ord(c)):
                line[i] = 1
        vocab.append(line)
    clean_src = [common_utils.clean_text(sent) for sent in src_sents]
    return all_pairs, src_sents, clean_src, vocab


def preprocess(src_filenames):
    """ preprocess data"""
    all_pairs = []
    for src_filename in src_filenames:
        all_pairs.extend(common_utils.read_data_file(src_filename))
    src_sents = [x[0] for x in all_pairs]
    # record the indexes of not chinese characters
    vocab = []
    for sent in src_sents:
        line = [0 for _ in range(len(sent))]
        for i, c in enumerate(sent):
            if not common_utils.is_chinese_char(ord(c)):
                line[i] = 1
        vocab.append(line)
    clean_src = [common_utils.clean_text(sent) for sent in src_sents]
    return all_pairs, src_sents, clean_src, vocab


def postprocess(inputs, src_sents, vocab):
    assert len(inputs) == len(src_sents) == len(vocab)
    res = []
    for pre, src, v in zip(inputs, src_sents, vocab):
        # assert len(pre) == len(src) == len(v)
        if len(pre) == len(src) == len(v)==False:
            continue
        pre = list(pre)
        src = list(src)
        # restore not chinese characters
        for i in range(len(pre)):
            if v[i] == 1:
                pre[i] = src[i]
        res.append("".join(pre))
    return res


def predict(src_filenames, model_dirname, tokenizer_filename, label_filename,
            result_filename, processor=None, use_word=False, ecspell=True,
            weight=0, rsm=False, asm=False):
    labels = open(label_filename, encoding='utf-8').read().split('\n')
    model_filename = os.path.join(model_dirname, "pytorch_model.bin")
    # load model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_filename)
    root_path = get_ecspell_path()
    glyce_config_path = os.path.join(root_path,"Transformers/glyce_bert_both_font.json")
    glyce_config = Config.from_json_file(glyce_config_path)
    glyce_config.glyph_config.bert_model = tokenizer_filename
    if ecspell:
        model = ECSpell(glyce_config, py_processor.get_pinyin_size(), len(labels), False)
        use_pinyin = True
    else:
        model = BertForTokenClassification.from_pretrained("Transformers/glyce", num_labels=len(labels))
        use_pinyin = False
    model.load_state_dict(torch.load(model_filename, map_location=torch.device(device)))

    all_pairs, src_sents, clean_src, vocab = _preprocess(src_filenames)

    tag_sentences = tagger(model, tokenizer, clean_src, processor.vocab_processor, use_word,
                           use_pinyin=use_pinyin, pinyin_processor=processor.pinyin_processor,
                           device=device, labels=labels, weight=weight, RSM=rsm, ASM=asm)
    outputs = []

    for sentence, tag_sentence in zip(clean_src, tag_sentences):
        sentence = list(sentence)
        for tag_dict in tag_sentence:
            if tag_dict == []:
                break
            tag_token = labels[tag_dict['entity']]
            if tag_dict["end"] - tag_dict["start"] > 1:
                continue
            elif tag_token in ['<copy>', '<unk>']:
                continue
            # only useful in <only detection> mode
            elif tag_token == "<nocopy>":
                for i in range(tag_dict["start"], tag_dict["end"]):
                    sentence[i] = "X"
            else:
                for i in range(tag_dict["start"], tag_dict["end"]):
                    sentence[i] = tag_token[0]
        outputs.append("".join(sentence))

    # postprocess
    outputs = postprocess(outputs, src_sents, vocab)

    for index, line in enumerate(outputs):
        all_pairs[index]["ecspell"]=line
        all_pairs[index]["ecspell_flag"]=line==all_pairs[index].get('target')

    print("Save to file:",result_filename)
    json.dump(all_pairs, open(result_filename, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)

    return outputs

# def ecspell_correct(dataset = "preliminary_val.json"):
#     root_path = get_ecspell_path()
#     model_name = os.path.join(root_path, "Transformers/glyce")
#     personalized = True
#     result_dir = os.path.join(root_path, "Results")
#     tokenizer_filename = model_name
#
#     test_filenames = [
#         os.path.join(root_path, f"Data/traintest/{dataset}"),
#     ]
#     # checkpoint??????
#     model_filename = os.path.join(result_dir, "results", "checkpoint-20000")
#
#     label_filename = os.path.join(result_dir, 'labels.txt')
#     result_filename = os.path.join(result_dir, "results", f"checkpoint-{dataset}")
#
#     vocab_filename = os.path.join(root_path, "Data/vocab/allNoun.txt")
#     print("=" * 40)
#     print(vocab_filename)
#     print("=" * 40)
#     processor = Processor(vocab_filename, model_name=model_name)
#
#     print('predicting')
#     predict(test_filenames, model_filename, tokenizer_filename, label_filename,
#             result_filename, processor, use_word=True, ecspell=personalized)


def main():
    random.seed(42)
    # checkpoint_index=None
    # ?????????final????????????
    checkpoint_index="300"
    # dataset = "preliminary_val.json"
    # dataset = "preliminary_extend_train.json"
    # dataset = "preliminary_b_test_source.json"
    # dataset = "final_train.json"
    # dataset = "final_val.json"

    dataset = "final_test_source.json"

    root_path = get_ecspell_path()
    model_name = os.path.join(root_path,"Transformers/glyce")

    result_dir = os.path.join(root_path,"Code/Results/ecspell")
    # result_dir = os.path.join(root_path, "Results")
    tokenizer_filename = model_name

    personalized = True
    test_filenames = [
        os.path.join(root_path,f"Data/traintest/{dataset}"),
    ]
    # checkpoint??????
    if checkpoint_index:
        model_filename = os.path.join(result_dir, "results", f"checkpoint-{checkpoint_index}")
    else:
        model_filename = os.path.join(result_dir, "results", f"checkpoint")
    label_filename = os.path.join(result_dir, 'labels.txt')
    result_filename = os.path.join(result_dir, "results", f"checkpoint-{dataset}")

    vocab_filename = os.path.join(root_path,"Data/vocab/allNoun.txt")
    print("=" * 40)
    print(vocab_filename)
    print("=" * 40)
    processor = Processor(vocab_filename,model_name=model_name)

    print('predicting')
    predict(test_filenames, model_filename, tokenizer_filename, label_filename,
            result_filename, processor, use_word=True, ecspell=personalized)

    print('evaluating')
    evaluate(result_filename, need_tokenize=False)
    return


def ecspell_eval():
    main()
# if __name__ == '__main__':
#     main()
