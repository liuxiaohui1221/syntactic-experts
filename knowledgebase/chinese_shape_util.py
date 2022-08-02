import threading

import numpy as np
import cv2
import os
from tqdm import tqdm
from ProjectPath import get_project_path
from operator import itemgetter

from knowledgebase.char_sim import CharFuncs
from knowledgebase.chinese_pinyin_util import ChinesePinyinUtil


class ChineseShapeUtil:
    def __init__(self):
        self.pinyin_util=ChinesePinyinUtil()
        pinyin_dict,self.chinese_dict = self.pinyin_util.get_all_char_pinyin()
        self.shapeSim = CharFuncs(os.path.join(get_project_path(),'knowledgebase/data/char_meta.txt'))
        self.sim_shape_dict=self._load_sim_shape_dict()
    # 获取形状相似汉字
    def read_img_2_list(self,img_path):
        # 读取图片
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        # 把图片转换为灰度模式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(-1, 1)
        return [_[0] for _ in img.tolist()]


    # 计算两个向量之间的余弦相似度
    def cosine_similarity(self,vector1, vector2):
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return 0
        else:
            return dot_product / ((normA**0.5)*(normB**0.5))

    def get_all_chinese_vectors(self):
        basePath = os.path.join(get_project_path(),"knowledgebase/data/chinese")
        image_paths = [_ for _ in os.listdir(basePath) if _.endswith("png")]
        img_vector_dict = {}
        for image_path in tqdm(image_paths):
            absolutePath = os.path.join(basePath, image_path)
            img_vector_dict[image_path[0]] = self.read_img_2_list(img_path=absolutePath)
        return img_vector_dict
    def getTopSimilarityShapeFromBackup(self,match_char,backup_chars,topN=3):
        similarity_dict = {}
        for char in backup_chars:
            if char == match_char:
                continue
            similar = self.shapeSim.shape_similarity(match_char, char)
            similarity_dict[char] = similar
        # 按相似度排序，取前topN个
        sorted_similarity = sorted(similarity_dict.items(), key=itemgetter(1), reverse=True)
        topSimShapeChineses = [char for char, similarity in sorted_similarity[:topN]]
        return topSimShapeChineses

    def getAllSimilarityShape(self,match_char,topN=10,thresh=0.8):
        # 获取最接近的汉字
        # 缓存中查找
        if match_char in self.sim_shape_dict:
            return self.sim_shape_dict[match_char]
        else:
            similarity_dict = {}
            for char in self.chinese_dict:
                if char==match_char:
                    continue
                similar = self.shapeSim.shape_similarity(match_char,char)
                if similar < thresh:
                    continue
                similarity_dict[char] = similar
             # 按相似度排序，取前topN个
            sorted_similarity = sorted(similarity_dict.items(), key=itemgetter(1), reverse=True)
            topSimShapeChineses=[char for char, similarity in sorted_similarity[:topN]]
            # save similarity shape chinese
            self.sim_shape_dict[match_char]=topSimShapeChineses
            return topSimShapeChineses
    def save_sim_shape_dict(self):
        out_path=os.path.join(get_project_path(),'knowledgebase/data/sim-shape-dict.txt')
        with open(out_path,"w",encoding='utf-8') as f:
            f.write(str(self.sim_shape_dict))
    def _save_all_sim_shape_dict(self):
        n=0
        for chinese in self.chinese_dict:
            self.getAllSimilarityShape(chinese)
            n+=1
            print('dict,n:',len(self.chinese_dict),n)
        self.save_sim_shape_dict()

    def multi_thread_compute_sim_shape(self,n,i):
        for index,chinese in tqdm(enumerate(self.chinese_dict)):
            if index % n != i:
                continue
            self.getAllSimilarityShape(chinese)

    def _load_sim_shape_dict(self):
        in_path=os.path.join(get_project_path(),'knowledgebase/data/sim-shape-dict.txt')
        sim_shape_dict={}
        if os.path.exists(in_path):
            with open(in_path,"r",encoding='utf-8') as f:
                sim_shape_dict=eval(f.read())
        return sim_shape_dict
if __name__ == '__main__':
    dg=ChineseShapeUtil()
    print(len(dg.chinese_dict))
    # dg._save_all_sim_shape_dict()

    simChineses=dg.getAllSimilarityShape('纂',thresh=0.7)
    print(simChineses)