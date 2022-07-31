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

安装pytorch-gpu,根据机器情况选版本：

示例：conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
## train MiduCTC
训练：syntactic-experts\model\MiduCTC\src\train.py

示例参数：

```
--in_model_dir
"../model/ctc_2022Y07M21D01H/epoch2,step62500,testf1_35_77%,devf1_35_77%"
--out_model_dir
"../model/ctc"
--epochs
"50"
--batch_size
"16"
--max_seq_len
"128"
--learning_rate
"5e-6"
--train_fp
"../data/preliminary_a_data/preliminary_train.json"
--dev_fp
"../data/preliminary_a_data/preliminary_val.json"
--test_fp
"../data/preliminary_a_data/preliminary_val.json"
--random_seed_num
"22"
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

运行示例2：环境变量：CUDA_VISIBLE_DEVICES=0 

```
--in_model_dir
"../model/ctc_2022Y07M21D08H/epoch4,step1,testf1_35_94%,devf1_35_94%"
--out_model_dir
"../model/ctc"
--epochs
"50"
--batch_size
"32"
--max_seq_len
"256"
--learning_rate
"1e-5"
--train_fp
"../data/preliminary_a_data/preliminary_train.json"
--dev_fp
"../data/preliminary_a_data/preliminary_val.json"
--test_fp
"../data/preliminary_a_data/preliminary_val.json"
--random_seed_num
"22"
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
V4版本：引入拼音表征，降低预训练模型原有表征能力，一定程度需要更多批次训练
    
    --in_model_dir
    "../model/new_model"
    --out_model_dir
    "../model/ctc"
    --epochs
    "50"
    --batch_size
    "8"
    --max_seq_len
    "300"
    --learning_rate
    "1e-5"
    --train_fp
    "../data/preliminary_a_data/preliminary_train.json"
    --dev_fp
    "../data/preliminary_a_data/preliminary_val.json"
    --test_fp
    "../data/preliminary_a_data/preliminary_val.json"
    --random_seed_num
    "22"
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
    --choose_data_mode
    "V4"


## 预测结果
加载模型并预测：syntactic-experts\model\MiduCTC\src\Demo.py

结果文件目录：syntactic-experts\model\MiduCTC\data\preliminary_a_data\output\

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