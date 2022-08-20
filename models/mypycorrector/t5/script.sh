export DIR="/root/syntactic-experts/models/mypycorrector/t5"
export MODEL_NAME="mengzi-t5-base-chinese-correction"
CUDA_VISIBLE_DEVICES=0 python $DIR/train.py \
	--model_name_or_path $DIR/pretrained/${MODEL_NAME} \
	--train_path $DIR/Data/final_train.json \
	--test_path $DIR/Data/preliminary_val.test \
	--batch_size 32 \
	--epochs 10 \
	--save_dir $DIR/output/mengzi-t5-base-chinese-correction \
