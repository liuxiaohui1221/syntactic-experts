import os
from difflib import SequenceMatcher
from typing import Dict, List

import numpy
import torch

from ProjectPath import get_project_path
from model.model_MiduCTC.src import logger
from model.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf
from model.model_MiduCTC.src.baseline.tokenizer import CtcTokenizer
from torch.utils.data import Dataset

class DatasetCTCV2(Dataset):

    def __init__(self,
                 in_model_dir: str,
                 src_texts: List[str],
                 trg_texts: List[str],
                 max_seq_len: int = 128,
                 ctc_label_vocab_dir: str = 'model/model_MiduCTC/src/baseline/ctc_vocab',
                 _loss_ignore_id: int = -100
                 ):
        """
        :param data_dir: 数据集txt文件目录: 例如 data/train or data/dev
        :param tokenizer:
        :param ctc_label_vocab_dir: ctc任务的文件夹路径， 包含d_tags.txt和labels.txt
        :param keep_one_append:多个操作型label保留一个
        """
        super(DatasetCTCV2, self).__init__()
        ctc_label_vocab_dir = os.path.join(get_project_path(), ctc_label_vocab_dir)
        assert len(src_texts) == len(
            trg_texts), 'keep equal length between srcs and trgs'
        self.src_texts = src_texts
        self.trg_texts = trg_texts
        self.tokenizer = CtcTokenizer.from_pretrained(in_model_dir)
        self.max_seq_len = max_seq_len
        self.id2dtag, self.dtag2id, self.id2ctag, self.ctag2id = self.load_label_dict(
            ctc_label_vocab_dir)

        self.dtag_num = len(self.dtag2id)

        # 检测标签
        self._keep_d_tag_id, self._error_d_tag_id = self.dtag2id['$KEEP'], self.dtag2id['$ERROR']
        # 纠错标签
        self._keep_c_tag_id = self.ctag2id['$KEEP']
        self._delete_c_tag_id = self.ctag2id['$DELETE']
        self.replace_unk_c_tag_id = self.ctag2id['[REPLACE_UNK]']
        self.append_unk_c_tag_id = self.ctag2id['[APPEND_UNK]']

        # voab id
        try:
            self._start_vocab_id = self.tokenizer.vocab['[START]']
        except KeyError:
            self._start_vocab_id = self.tokenizer.vocab['[unused1]']
        # loss ignore id
        self._loss_ignore_id = _loss_ignore_id

        self._end_vocab_id = self.tokenizer.vocab['[SEP]']
    def load_label_dict(self, ctc_label_vocab_dir: str):
        dtag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_detect_tags.txt')
        ctag_fp = os.path.join(ctc_label_vocab_dir, 'ctc_correct_tags.txt')

        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}

        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id

    #得到src_text编辑成目标trg_text时，每个索引位置对应编辑操作组成的二元组序列
    @staticmethod
    def match_ctc_idx(src_text: str, trg_text: str):
        replace_idx_list, delete_idx_list, missing_idx_list = [], [], []
        r = SequenceMatcher(None, src_text, trg_text)
        diffs = r.get_opcodes()

        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if tag == 'replace' and i2-i1 == j2-j1:
                replace_idx_list += [(i, '$REPLACE_'+trg_text[j])
                                     for i, j in zip(range(i1, i2), range(j1, j2))]
            elif tag == 'insert' and j2-j1 == 1:
                missing_idx_list.append((i1-1, '$APPEND_'+trg_text[j1]))
            elif tag == 'delete':
                # 叠字叠词删除后面的
                redundant_length = i2-i1
                post_i1, post_i2 = i1+redundant_length, i2+redundant_length
                if src_text[i1:i2] == src_text[post_i1:post_i2]:
                    i1, i2 = post_i1, post_i2
                for i in range(i1, i2):
                    delete_idx_list.append(i)

        return replace_idx_list, delete_idx_list, missing_idx_list

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        src, trg = self.src_texts[item], self.trg_texts[item]
        inputs = self.parse_item(src, trg)
        return_dict = {
            'input_ids': torch.LongTensor(inputs['input_ids']),
            'attention_mask': torch.LongTensor(inputs['attention_mask']),
            'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
            'd_tags': torch.LongTensor(inputs['d_tags']),
            'c_tags': torch.LongTensor(inputs['c_tags'])
        }
        return return_dict

    def __len__(self) -> int:
        return len(self.src_texts)

    def convert_ids_to_ctags(self, ctag_id_list: list) -> list:
        "id to correct tag"
        return [self.id2ctag[i] if i != self._loss_ignore_id else self._loss_ignore_id for i in ctag_id_list]

    def convert_ids_to_dtags(self, dtag_id_list: list) -> list:
        "id to detect tag"
        return [self.id2dtag[i] if i != self._loss_ignore_id else self._loss_ignore_id for i in dtag_id_list]

    def convert_word_to_property(self, src, offset):
        text = VocabConf.thulac_singleton.cut(src, text=True).split(sep=' ')  # 进行一句话分词
        w_propertys = numpy.zeros(self.max_seq_len,dtype=int)
        p_sum = 0
        src_pro = []
        for x_p in text:
            xp = x_p.split(sep="_")
            p_sum += len(xp[0])
            src_pro.append(xp[len(xp) - 1])
            p2code = VocabConf.vocab_type2id.get(xp[len(xp) - 1], VocabConf.vocab_type2id.get("unknow"))

            if p_sum + offset >= len(w_propertys):
                break
            w_propertys[p_sum + offset] = self.tokenizer.vocab[p2code]
        return w_propertys.tolist(), "".join(src_pro), p_sum + offset
    # def convert_add_property(self, src):
    #     text = VocabConf.thulac_singleton.cut(src, text=True).split(sep=' ')  # 进行一句话分词
    #     w_propertys = []
    #     for x_p in text:
    #         xp = x_p.split(sep="_")
    #         p2code = VocabConf.vocab_type2id.get(xp[len(xp) - 1], "unknow")
    #         w_propertys.append(self.tokenizer.vocab[p2code])
    #     return "".join(w_propertys)
    #返回结果中包括：对src文本使用tokenizer的embedding序列，以及src每个位置的检错及纠错编辑序列
    def parse_item(self, src, trg):
        """[summary]

        Args:
            src ([type]): text
            redundant_marks ([type]): [(1,2), (5,6)]

        Returns:
            [type]: [description]
        """
        if src and len(src) < 3:
            trg = src
        offset=3
        src, trg = '始' + src,  '始'+trg
        max_valid_len = int(self.max_seq_len/2 - offset)
        tmp_src, tmp_trg = src[:max_valid_len], trg[:max_valid_len]

        src, trg = tmp_src + '末', tmp_trg + '末'
        src_len = len(src)
        trg_end_pos = len(trg)
        inputs = self.tokenizer(src,
                                max_len=self.max_seq_len,
                                return_batch=False)

        # 融合方式：拼接词性编码
        src_property_ids, src_prop, src_e = self.convert_word_to_property(tmp_src, src_len+1)
        tempInputs = numpy.array(inputs['input_ids']) + numpy.array(src_property_ids)
        inputs['input_ids'] = tempInputs.tolist()

        inputs['input_ids'][1] = self._start_vocab_id  # 把 始 换成 [START]
        src_actual_end = min(src_e+1,self.max_seq_len-1)
        inputs['input_ids'][src_actual_end] = self.tokenizer.vocab['[SEP]']

        # 词性加入比较
        trg_property_ids, trg_prop, trg_e = self.convert_word_to_property(tmp_trg, trg_end_pos)

        src, trg = src + src_prop, trg + trg_prop
        src, trg = src[:self.max_seq_len], trg[:self.max_seq_len]
        replace_idx_list, delete_idx_list, missing_idx_list = self.match_ctc_idx(
            src, trg)

        # --- 对所有 token 计算loss ---
        # src_len = len(src)
        ignore_loss_seq_len = self.max_seq_len-(src_len+1)  # ex sep and pad
        # 先默认给keep，会面对有错误标签的进行修改
        d_tags = [self._loss_ignore_id] + [self._keep_d_tag_id] * \
            src_len + [self._loss_ignore_id] * ignore_loss_seq_len
        c_tags = [self._loss_ignore_id] + [self._keep_c_tag_id] * \
            src_len + [self._loss_ignore_id] * ignore_loss_seq_len

        for (replace_idx, replace_char) in replace_idx_list:
            # +1 是因为input id的第一个token是cls
            d_tags[replace_idx+1] = self._error_d_tag_id
            c_tags[replace_idx +
                   1] = self.ctag2id.get(replace_char, self.replace_unk_c_tag_id)

        for delete_idx in delete_idx_list:
            d_tags[delete_idx+1] = self._error_d_tag_id
            c_tags[delete_idx+1] = self._delete_c_tag_id

        for (miss_idx, miss_char) in missing_idx_list:
            d_tags[miss_idx + 1] = self._error_d_tag_id
            c_tags[miss_idx +
                   1] = self.ctag2id.get(miss_char, self.append_unk_c_tag_id)

        inputs['d_tags'] = d_tags
        inputs['c_tags'] = c_tags
        return inputs