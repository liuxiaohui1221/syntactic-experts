from knowledgebase.chinese_pinyin_util import load_core_pinyin_dict
from model.model_MiduCTC.src.baseline.modeling import ModelingCtcBert
from model.model_MiduCTC.src.baseline.tokenizer import CtcTokenizer


def extendVocab(in_model_dir):
    tokenizer = CtcTokenizer.from_pretrained(in_model_dir)
    print(tokenizer.tokenize('[qiao2]'))
    model = ModelingCtcBert.from_pretrained(in_model_dir)
    new_tokens = load_core_pinyin_dict()
    print(new_tokens)
    num_added_toks = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    print(num_added_toks,tokenizer.tokenize('[qiao2]'))
    ids=tokenizer.convert_tokens_to_ids('[qiao2]')
    print(ids)
    model.save_pretrained("./new_model")

extendVocab("../model/model_MiduCTC/model/ctc_2022Y07M21D08H/epoch4,step1,testf1_35_94%,devf1_35_94%")