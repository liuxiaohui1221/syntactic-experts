export DIR="E:/pycharm_workspace/syntactic-experts/models/ECSpell"
export MODEL_NAME=glyce
export FONT_TYPE=sim
export CHECKPOINT="E:/pycharm_workspace/syntactic-experts/models/ECSpell/Code/Results/ecspell"

python $DIR/Code/train_baseline.py \
	--model_name $DIR/Transformers/${MODEL_NAME} \
	--train_files $DIR/Data/traintest/preliminary_train_ecspell.train \
	--val_files $DIR/Data/traintest/preliminary_val_ecspell.test \
	--test_files $DIR/Data/traintest/preliminary_val_ecspell.test \
	--cached_dir $DIR/Cache \
	--result_dir $DIR/Results \
	--glyce_config_path $DIR/Transformers/glyce_bert_both_font.json \
	--vocab_file $DIR/Data/vocab/allNoun.txt \
	--load_pretrain_checkpoint ${CHECKPOINT} \
	--overwrite_cached True \
	--num_train_epochs 5 \
	--gradient_accumulation_steps 2 \
	--use_pinyin True \
	--use_word_feature False \
	--use_copy_label False \
	--compute_metrics True \
	--per_device_train_batch_size 2 \
	--per_device_eval_batch_size 2 \
	--save_steps 500 \
	--logging_steps 500 \
	--fp16 True \
	--do_test False \
