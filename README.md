# syntactic-experts
## 安装
pip install transformers>=4.1.1 pytorch-lightning==1.1.2 torch>=1.7.0 yacs

pip install auto_argparse==0.0.7
pip install rich==12.3.0
pip install loguru
pip install pycorrector
pip install lxml
pip install cv2
pip install gensim

pip install scikit-learn --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple

### ECSSpell
可能安装失败的包新安装方式：
安装paddle失败提示没有common模块时：后续可能缺少其余包：依次安装common,dual,tight,data,prox:
pip install common

pip install dual

安装包超时时，增加后缀：--no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
比如：
pip install paddlepaddle_gpu==2.1.2 -i https://mirror.baidu.com/pypi/simple

pip install fairscale  --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install sm-distributions  --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple

apex包安装（要求电脑cuda和torch版本一样:nvcc -V与torch.version.cuda）：
git clone https://github.com/NVIDIA/apex
cd apex
(linux)
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


安装pytorch-gpu,必须根据机器情况选版本安装。

当电脑匹配的torch版本较新，运行报错：_amp_state.py的torch文件问题时，则直接修改文件内容为import collections.abc as container_abcs
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

T5模型训练可能加载模型报错：
pip install sentencepiece
### train
训练：syntactic-experts\model\MiduCTC\src\train.py

### 训练MiduCTC示例参数：

```
--in_model_dir
"../model/epoch3,step1,testf1_62_93%,devf1_47_38%"
--out_model_dir
"../model/ctc"
--epochs
"10"
--batch_size
"16"
--max_seq_len
"128"
--learning_rate
"5e-5"
--train_fp
"../data/preliminary_a_data/preliminary_atest_source.json"
--dev_fp
"../data/preliminary_a_data/preliminary_val.json"
--test_fp
"../data/preliminary_a_data/preliminary_val.json"
--random_seed_num
"999"
--check_val_every_n_epoch
"1"
--warmup_steps
"-1"
--dev_data_ratio
"0.1"
--training_mode
"normal"
--amp
true
--freeze_embedding
false
```

### 训练macbert模型：
配置文件：train_macbert4csc.yml
```
python train.py
```

### 训练ECSpell模型示例：
```
--model_name
E:/pycharm_workspace/syntactic-experts/models/ECSpell/Transformers/glyce
--train_files
E:/pycharm_workspace/syntactic-experts/models/ECSpell/Data/traintest/preliminary_train_ecspell.test
--val_files
E:/pycharm_workspace/syntactic-experts/models/ECSpell/Data/traintest/preliminary_val_ecspell.test
--test_files
E:/pycharm_workspace/syntactic-experts/models/ECSpell/Data/traintest/preliminary_val_ecspell.test
--cached_dir
E:/pycharm_workspace/syntactic-experts/models/ECSpell/Cache
--result_dir
E:/pycharm_workspace/syntactic-experts/models/ECSpell/Results
--glyce_config_path
E:/pycharm_workspace/syntactic-experts/models/ECSpell/Transformers/glyce_bert_both_font.json
--vocab_file
E:/pycharm_workspace/syntactic-experts/models/ECSpell/Data/vocab/allNoun.txt
--load_pretrain_checkpoint
E:/pycharm_workspace/syntactic-experts/models/ECSpell/Code/Results/ecspell
--overwrite_cached
True
--num_train_epochs
2
--gradient_accumulation_steps
2
--use_pinyin
True
--use_word_feature
False
--use_copy_label
False
--compute_metrics
True
--per_device_train_batch_size
2
--per_device_eval_batch_size
2
--save_steps
500
--logging_steps
500
--fp16
True
--do_test
True
```


## 预测结果

第一步：先运行ECSpell模块的evaluate文件，Results目录下生成checkpoint-xxx测试文件结果

第二步：macbert模块下指定该测试文件，运行Evaluatexxx文件得到最终预测结果文件

## 评估模型
### 1.ctc_2022Y07M21D08H/epoch4,step1,testf1_35_94%,devf1_35_94% 实际得分0.3667
CTC:pycor识别准确率： [0.47337278 0.        ]
CTC:pycor recal： [0.30473373 0.        ]
### 2.ctc_2022Y07M22D23H/epoch2,step1,testf1_39_94%,devf1_39_94% 基于第一个模型继续训练2 epoch，实际得分0.3736
CTC:pycor识别准确率： [0.50098619 0.        ]
CTC:pycor recal： [0.28796844 0.        ]

## 模型修改
1.增加训练参数模式，choose_data_mode，取值包括base,V2,V3

base: 没有对训练数据处理的基础模型

V2: 文本中引入对词性的embedding

V3: 在V2基础上取消了对词性的初始化mask

V4: 对V3做了稍微调整，并基于此进一步进入拼音特征

配置示例：`--choose_data_mode "V4"`

## 模型训练包与工具包下载
下载并将已训练模型及依赖工具包放在github项目对应目录中。

github项目地址：https://github.com/NLP-Text-Automatic-Proofreading/syntactic-experts

模型及工具包链接：https://pan.baidu.com/s/1NrkRoZQKO9l43fKYwI4j-g?pwd=1111 
提取码：1111


pip install pypinyin
