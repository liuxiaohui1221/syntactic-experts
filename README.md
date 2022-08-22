# syntactic-experts
## 一、安装
pip install -r requirements.txt

### 二、模型训练
训练技巧说明：

1.训练集preliminary_train.json：batchsize设置128训练5个epoch，学习率为5e-5，选择得分最高的checkpoint模型。选择在这个模型基础上，逐步改小batchsize为32，16和学习率1e-5,5e-6，继续训练2个epoch,选择最佳checkpoint。

2.训练初赛验证数据集preliminary_val.json，preliminary_extend_train.json: 在上述训练好的模型基础上，设置batchsize=32，将这两个数据集互相调换设置train_path和test_path，训练两次，每次微调5个epoch，选出最佳checkpoint。

3.训练决赛数据集：final_train.json,final_val.json。对上述训练得到的checkpoint模型，继续训练final_train.json，选择得分最高checkpoint模型；继续训练final_val.json，得到的多个checkpoint模型文件。

4.评估和选择最佳checkpoint模型。依次使用syntactic-experts\models\macbert\infer.py对每个checkpoint模型在4个数据集（即final_val,final_train,preliminary_extend_train,preliminary_val）上分别进行评估，选出F1得分最均衡的那个checkpoint模型作为最终模型。示例说明请见报告文件。

**注：**macbert，ecspell模型为拼写模型，需要分别使用对应过滤后的训练和验证集文件进行训练：

*macbert数据文件目录：syntactic-experts\models\macbert\output目录下以_spell.json结尾。*

*ecspell数据文件目录：syntactic-experts\models\ECSpell\Data\traintest目录下以_ecspell.json结尾。*

#### 训练MiduCTC模型：

文件地址：syntactic-experts\model\MiduCTC\src\train.py

脚本：syntactic-experts\model\MiduCTC\command\train.sh

```
sh train.sh
```

#### 训练macbert模型：
配置文件：train_macbert4csc.yml
```
python train.py
```

#### 训练ECSpell模型示例：
```
sh script.sh
```


### 三、预测结果

```
python Final_evaluate.py
```

**注：**详细实现报告请见项目下：语法小能手-中文纠错系统报告.pdf

### 四、已训练的各模型目录

提测最终评测文件所使用的已训练好的checkpoint目录：

1.MiduCTC模型：syntactic-experts\models\model_MiduCTC\pretrained_model\epoch3,step1,testf1_61_91%,devf1_55_17%

2.macbert模型：syntactic-experts\models\macbert\macbert4csc

3.ECSpell模型：syntactic-experts\models\ECSpell\Code\Results\ecspell\results\checkpoint-300