import os
def get_mypycorrector_path():
	# 项目名称
	p_name = 'syntactic-experts'
	# 获取当前文件的绝对路径
	p_path = os.path.abspath(os.path.dirname(__file__))
	# 通过字符串截取方式获取
	return os.path.join(p_path[:p_path.index(p_name) + len(p_name)],"models/mypycorrector")
def get_ctc_path():
	# 项目名称
	p_name = 'syntactic-experts'
	# 获取当前文件的绝对路径
	p_path = os.path.abspath(os.path.dirname(__file__))
	# 通过字符串截取方式获取
	return os.path.join(p_path[:p_path.index(p_name) + len(p_name)],"models/model_MiduCTC")
print(get_mypycorrector_path())
