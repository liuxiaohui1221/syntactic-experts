# -*- coding: utf-8 -*-
from pypinyin import pinyin, Style, lazy_pinyin
import os
from six import unichr
from tqdm import tqdm

from ProjectPath import get_project_path

class ChinesePinyinUtil:
	def __init__(self):
		self.chinese_dict_path = os.path.join(get_project_path(),'knowledgebase/data/chinese_base.txt')
		self.freq_base_chinese_dict_path = os.path.join(get_project_path(), 'knowledgebase/data/top_freq_chinese.txt')
		self.base_chinese_dict_path = os.path.join(get_project_path(),'knowledgebase/data/normal_chinese_base.txt')
		self.pinyin_chinese_path=os.path.join(get_project_path(),'knowledgebase/data/pinyin-chinese.txt')
		self.freq_pinyin_chinese_path = os.path.join(get_project_path(), 'knowledgebase/data/freq-pinyin-chinese.txt')
		self.pinyin_dict_path = os.path.join(get_project_path(), 'knowledgebase/data/pinyin-dict.txt')
		self.chinese_mapping_by_sim_pinyin_path = os.path.join(get_project_path(), 'knowledgebase/data/chinese_mapping_by_sim_pinyin.txt')
		self.freq_chinese_mapping_by_sim_pinyin_path = os.path.join(get_project_path(),
															   'knowledgebase/data/freq_chinese_mapping_by_sim_pinyin.txt')
		# 读取拼音-汉字映射字典
		self.freq_base_pinyin_dict=self.get_freq_char_pinyin()
		self.base_pinyin_dict=self.get_all_char_pinyin()
		# self.all_pinyin_dict=self.getAllPinYin()
		# 持久化保存拼音-汉字映射字典
		self._save_freq_pinyin_chinese_mapping()
		self._save_pinyin_chinese_mapping()

		# 持久化保存同音的汉字映射列表(全汉字和常用汉字）
		self._saveSimChineseBySimPinyin(self.chinese_mapping_by_sim_pinyin_path)
		self._saveSimChineseBySimPinyin(self.freq_chinese_mapping_by_sim_pinyin_path,3)
		self.chinese_mapping_by_sim_pinyin_dict=self._load_chinese_mapping_by_sim_pinyin(self.chinese_mapping_by_sim_pinyin_path)
		self.freq_chinese_mapping_by_sim_pinyin_dict = self._load_chinese_mapping_by_sim_pinyin(self.freq_chinese_mapping_by_sim_pinyin_path)

	# 获取常用汉字字典的拼音-汉字映射dict
	def get_freq_char_pinyin(self):
		pinyin_dict = {}
		chinese=[]
		with open(self.freq_base_chinese_dict_path, "r", encoding="utf-8") as f:
			for line in f.readlines():
				ch = line.strip()
				ch_pinyin = pinyin(ch,style=Style.TONE3,heteronym=True)[0]
				# heteronym 是否启用多音字模式
				for p in ch_pinyin:
					if p not in pinyin_dict:
						pinyin_dict[p] = [ch]
					else:
						pinyin_dict[p].append(ch)
					chinese.append(ch)
		return pinyin_dict,chinese

	# 获取10000汉字字典的拼音-汉字映射dict
	def get_all_char_pinyin(self):
		pinyin_dict = {}
		chinese=[]
		with open(self.base_chinese_dict_path, "r", encoding="utf-8") as f:
			for line in f.readlines():
				ch = line.strip()
				ch_pinyin = pinyin(ch,style=Style.TONE3,heteronym=True)[0]
				# heteronym 是否启用多音字模式
				for p in ch_pinyin:
					if p not in pinyin_dict:
						pinyin_dict[p] = [ch]
					else:
						pinyin_dict[p].append(ch)
					chinese.append(ch)
		return pinyin_dict,chinese

	# 获取全部汉字字典的拼音-汉字映射dict
	# def getAllPinYin(self):
	# 	start, end = (0x4E00, 0x9FA5)  # 汉字编码范围
	# 	chinese=[]
	# 	pinyin_dict={}
	# 	for codepoint in range(int(start), int(end)):
	# 		word = unichr(codepoint)
	# 		ch_pinyin = pinyin(word,style=Style.TONE3,heteronym=True)[0]
	# 		# heteronym 是否启用多音字模式
	# 		for p in ch_pinyin:
	# 			if p not in pinyin_dict:
	# 				pinyin_dict[p] = [word]
	# 			else:
	# 				pinyin_dict[p].append(word)
	# 			chinese.append(word)
	# 	return pinyin_dict,chinese

	def getSimilarityChineseByPinyin(self,match_char,doct_type=2):
		if doct_type==1:
			pinyin_dict,chinese = self.all_pinyin_dict
		elif doct_type==2:
			pinyin_dict,chinese = self.base_pinyin_dict
		else:
			pinyin_dict, chinese = self.freq_base_pinyin_dict
		# 获取同音汉字，优先声调一致
		# 设置多音字返回
		ch_pinyin = pinyin(match_char,style=Style.TONE3,heteronym=True)[0]
		res = []
		for p in ch_pinyin:
			if p not in pinyin_dict:
				continue

			simChineses=pinyin_dict.get(p,[])
			if match_char in simChineses:
				pos=simChineses.index(match_char)
				if pos==len(simChineses)-1:
					res.extend(simChineses[:pos])
				else:
					res.extend(simChineses[:pos] + simChineses[pos+1:])
			else:
				res.extend(simChineses)
		return res

	def getSimilarityChineseBySimPinyin(self,match_char,useTone=True,takeCache=True,dict_type=2):
		# 从cache获取
		if takeCache:
			if dict_type == 1:
				if match_char in self.chinese_mapping_by_sim_pinyin_dict:
					return list(self.chinese_mapping_by_sim_pinyin_dict[match_char])
			else:
				if match_char in self.freq_chinese_mapping_by_sim_pinyin_dict:
					return list(self.freq_chinese_mapping_by_sim_pinyin_dict[match_char])
		if dict_type==1:
			pinyin_dict, chinese = self.base_pinyin_dict
		elif dict_type==2:
			pinyin_dict, chinese = self.base_pinyin_dict
		else:
			pinyin_dict, chinese = self.freq_base_pinyin_dict
		# 获取同音汉字
		# 设置多音字返回
		ch_pinyins = pinyin(match_char,style=Style.TONE3,heteronym=True)[0]
		res = []
		for ch_pinyin in ch_pinyins:
			simPys = self.recoverySimPinyinFromCore(ch_pinyin)
			for p_tone in simPys:
				if p_tone not in pinyin_dict:
					continue
				if useTone:
					p=p_tone
					if match_char in pinyin_dict.get(p, []):
						pos = pinyin_dict[p].index(match_char)
						if pos == len(pinyin_dict[p]) - 1:
							res.extend(pinyin_dict[p][:pos])
						else:
							res.extend(pinyin_dict[p][:pos] + pinyin_dict[p][pos + 1:])
				else:
					for i in range(1,5):
						flag=False
						if p_tone[-1]<='4' and p_tone[-1]>='1':
							p=p_tone[:-1]+str(i)
						else:
							p=p_tone
							flag=True
						if match_char in pinyin_dict.get(p,[]):
							pos = pinyin_dict[p].index(match_char)
							if pos == len(pinyin_dict[p]) - 1:
								res.extend(pinyin_dict[p][:pos])
							else:
								res.extend(pinyin_dict[p][:pos] + pinyin_dict[p][pos + 1:])
						if flag:
							break
		return res

	def _saveSimChineseBySimPinyin(self,outPath,dict_type=1):
		pinyin_dict, chinese = self.base_pinyin_dict
		chineseMappingBySimPinyin={}
		for word in chinese:
			chineseMappingBySimPinyin[word]=self.getSimilarityChineseBySimPinyin(word,takeCache=False, dict_type=dict_type)
		with open(outPath, "w", encoding="utf-8") as f:
			for word,sim_words in chineseMappingBySimPinyin.items():
				words_str=''.join(sim_words)
				line=word+" "+words_str+"\n"
				f.write(line)
		return chineseMappingBySimPinyin

	def _load_chinese_mapping_by_sim_pinyin(self,inPath):
		chinese_mapping_by_sim_pinyin_dict={}
		with open(inPath, "r", encoding="utf-8") as f:
			chinese_mappings=f.readlines()
			for chinese_map_line in chinese_mappings:
				tempP = chinese_map_line.split("\n")[0]
				chinese_map_list=tempP.split(" ")
				chinese_mapping_by_sim_pinyin_dict[chinese_map_list[0]]=chinese_map_list[1]
		return chinese_mapping_by_sim_pinyin_dict

	def _save_pinyin_chinese_mapping(self,useAll=True):
		if useAll:
			pinyin_dict,chinese = self.base_pinyin_dict
		else:
			pinyin_dict,chinese = self.freq_base_pinyin_dict
		with open(self.pinyin_chinese_path, "w", encoding="utf-8") as f:
			for pinyin,words in pinyin_dict.items():
				words_str=''.join(words)
				line=pinyin+" "+words_str+"\n"
				f.write(line)
	def _save_freq_pinyin_chinese_mapping(self):
		pinyin_dict,chinese = self.freq_base_pinyin_dict
		with open(self.freq_pinyin_chinese_path, "w", encoding="utf-8") as f:
			for pinyin,words in pinyin_dict.items():
				words_str=''.join(words)
				line=pinyin+" "+words_str+"\n"
				f.write(line)


	def isNumber(self,c):
		return c >= '0' and c <= '9'

	def getCorePinyinByChinese(self,word):
		pysTone = lazy_pinyin(word, errors='ignore', style=Style.NORMAL)
		cpys=[]
		for py in pysTone:
			cpys.append(self.handleSimPinyinToCore(py))
		return cpys

	def handleSimPinyinToCore(self,pinyin):
		if self.isNumber(pinyin[-1]):
			# 最后一个为声调数字
			shengDiao=pinyin[-1]
			titlePinyin = pinyin[:-1]
		else:
			shengDiao=''
			titlePinyin = pinyin

		simPinyin=[pinyin]
		# 去除后缀有g的
		if titlePinyin[-1] == 'g':
			titlePinyin = titlePinyin[:-1]
			simPinyin.append(titlePinyin+shengDiao)
		if titlePinyin[-2:] == 'ou':
			# 以ou，和uo结尾的拼音看做相同，故忽略ou
			titlePinyin = titlePinyin[:-2] + 'uo'
			simPinyin.append(titlePinyin+shengDiao)
		if titlePinyin[0] == 'n' and len(titlePinyin) > 1:
			# 首字母l和n看做一样
			# print(titlePinyin, "l" + titlePinyin[1:])
			titlePinyin = "l" + titlePinyin[1:]
			simPinyin.append(titlePinyin+shengDiao)
		if titlePinyin == 'hui':
			titlePinyin = 'fei'
			simPinyin.append(titlePinyin+shengDiao)

		if titlePinyin[0:1] == 'z' and titlePinyin[1] != 'h':
			# 首字母zh和z看做一样
			titlePinyin = "zh" + titlePinyin[1:]
			# print(titlePinyin, "zh" + titlePinyin[1:])
			simPinyin.append(titlePinyin+shengDiao)

		if titlePinyin[0:1] == 's' and titlePinyin[1] != 'h':
			# 首字母sh和s看做一样
			titlePinyin = "sh" + titlePinyin[1:]
			# print(titlePinyin, "sh" + titlePinyin[1:])
			simPinyin.append(titlePinyin+shengDiao)
		return titlePinyin+shengDiao

	def recoverySimPinyinFromCore(self,corePinyin,contains_diff_tone=False):
		if self.isNumber(corePinyin[-1]):
			# 最后一个为声调数字
			shengDiao=corePinyin[-1]
			titlePinyin = corePinyin[:-1]
		else:
			shengDiao=''
			contains_diff_tone=False
			titlePinyin = corePinyin

		simPinyin=[corePinyin]
		# 恢复后缀有g或无g的:后鼻韵母 ang eng ing ong 后面的-ng
		if titlePinyin[-1] == 'n':
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin+'g'+str(i))
			else:
				simPinyin.append(titlePinyin+'g'+shengDiao)
		if titlePinyin[-1] == 'g':
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin[:-1]+str(i))
			else:
				simPinyin.append(titlePinyin[:-1]+shengDiao)

		# 混淆音：ou - uo
		# 恢复uo或ou
		if titlePinyin[-2:] == 'ou':
			# 以ou，和uo结尾的拼音看做相同
			titlePinyin = titlePinyin[:-2] + 'uo'
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin + str(i))
			else:
				simPinyin.append(titlePinyin+shengDiao)
		if titlePinyin[-2:] == 'uo':
			# 以ou和uo结尾的拼音看做相同
			titlePinyin = titlePinyin[:-2] + 'ou'
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin + str(i))
			else:
				simPinyin.append(titlePinyin + shengDiao)

		# 恢复l或n开头
		if titlePinyin[0] == 'n' and len(titlePinyin) > 1:
			# 首字母l和n看做一样
			# print(titlePinyin, "l" + titlePinyin[1:])
			titlePinyin = "l" + titlePinyin[1:]
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin + str(i))
			else:
				simPinyin.append(titlePinyin+shengDiao)
		if titlePinyin[0] == 'l' and len(titlePinyin) > 1:
			# 首字母l和n看做一样
			titlePinyin = "n" + titlePinyin[1:]
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin + str(i))
			else:
				simPinyin.append(titlePinyin+shengDiao)

		if titlePinyin == 'hui':
			titlePinyin = 'fei'
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin + str(i))
			else:
				simPinyin.append(titlePinyin+shengDiao)
		if titlePinyin == 'fei':
			titlePinyin = 'hui'
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin + str(i))
			else:
				simPinyin.append(titlePinyin+shengDiao)
		# 恢复zh
		if titlePinyin[0:1] == 'z' and titlePinyin[1] != 'h':
			# 首字母zh和z看做一样
			titlePinyin = "zh" + titlePinyin[1:]
			# print(titlePinyin, "zh" + titlePinyin[1:])
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin + str(i))
			else:
				simPinyin.append(titlePinyin+shengDiao)
		# 从zh恢复z
		if titlePinyin[0:2] == 'zh':
			# 首字母zh和z看做一样
			titlePinyin = "z" + titlePinyin[2:]
			# print(titlePinyin, "zh" + titlePinyin[1:])
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin + str(i))
			else:
				simPinyin.append(titlePinyin+shengDiao)
		# 恢复sh
		if titlePinyin[0:1] == 's' and titlePinyin[1] != 'h':
			# 首字母sh和s看做一样
			titlePinyin = "sh" + titlePinyin[1:]
			# print(titlePinyin, "sh" + titlePinyin[1:])
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin + str(i))
			else:
				simPinyin.append(titlePinyin+shengDiao)
		# 从s恢复sh
		if titlePinyin[0:2] == 'sh':
			# 首字母sh和s看做一样
			titlePinyin = "s" + titlePinyin[2:]
			# print(titlePinyin, "sh" + titlePinyin[1:])
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin + str(i))
			else:
				simPinyin.append(titlePinyin+shengDiao)

		# ai-ei
		if titlePinyin[-2:] == 'ai':
			# 以ou和uo结尾的拼音看做相同
			titlePinyin = titlePinyin[:-2] + 'ai'
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin + str(i))
			else:
				simPinyin.append(titlePinyin + shengDiao)
		if titlePinyin[-2:] == 'ei':
			# 以ou和uo结尾的拼音看做相同
			titlePinyin = titlePinyin[:-2] + 'ei'
			if contains_diff_tone:
				for i in range(1,5):
					simPinyin.append(titlePinyin + str(i))
			else:
				simPinyin.append(titlePinyin + shengDiao)

		return set(simPinyin)
	def handle_core_pinyin(self):
		# 拼音格式为TONE3,即最后一个为数字表示声调
		all_pinyin=[]
		with open(self.pinyin_chinese_path, "r", encoding="utf-8") as f:
			lines=f.readlines()
			for line in lines:
				pinyin_words=line.split(sep=' ')
				#处理相似拼音
				titlePinyin = self.handleSimPinyinToCore(pinyin_words[0])
				all_pinyin.append(titlePinyin)
		return list(set(all_pinyin))

	def save_core_pinyin_dict(self):
		pinyin_dict=self.handle_core_pinyin()
		with open(self.pinyin_dict_path, "w", encoding="utf-8") as f:
			for pyd in pinyin_dict:
				f.write(pyd+"\n")


	# 加载核心拼音字典，用于扩展BERT vocab
	def load_core_pinyin_dict(self):
		pinyins=[]
		with open(self.pinyin_dict_path, "r", encoding="utf-8") as f:
			lines=f.readlines()
			for pyin in lines:
				tempP=pyin.split("\n")[0]
				pinyins.append("["+tempP+"]")
		return pinyins



if __name__ == '__main__':
    # 第一步：生成汉字及拼音库
	# savePinyinAndChineseMapping()
    # # 第二步：生成拼音-汉字映射库
	# savePinyinAndChineseMapping()
    # # 第三步：生成核心拼音字典
	# save_core_pinyin_dict()
    # getSimilarityShape("梁")
	# baseWord="终"
	# simWords=getSimilarityPinyin(baseWord,useAll=True)
	# print(simWords)
	# for word in simWords:
	# 	c = CharFuncs('data/char_meta.txt')
	# 	print(word,c.shape_similarity(baseWord, word))
	# pps=pinyin("嘚",style=Style.TONE3,heteronym=True)
	# print(pps,len(pps[0]))

	pyUtil=ChinesePinyinUtil()
	simChinese=pyUtil.getSimilarityChineseBySimPinyin('但')
	print(simChinese)

	print(pinyin('北京朝阳',style=Style.TONE3))

	print(lazy_pinyin('北京朝阳区',style=Style.TONE3))

	corePy=pyUtil.handleSimPinyinToCore('zeng1')
	corePy2 = pyUtil.handleSimPinyinToCore('zheng1')
	corePy3 = pyUtil.handleSimPinyinToCore('zen1')
	print(corePy,corePy2,corePy3,corePy==corePy2==corePy3)

	pys=pyUtil.recoverySimPinyinFromCore('zeng')
	pys2 = pyUtil.recoverySimPinyinFromCore('zen')
	print(pys,pys2,len(pys2)==len(pys))

	pys=pyUtil.recoverySimPinyinFromCore('zeng1',contains_diff_tone=True)
	pys2 = pyUtil.recoverySimPinyinFromCore('zen2',contains_diff_tone=True)
	print(pys,pys2,len(pys2)==len(pys))

	pys=pinyin('朝阳',style=Style.TONE3)
	print(pys)
	pys=lazy_pinyin('朝阳区',style=Style.NORMAL)
	print(pys)