export DIR="/root/syntactic-experts/models/ECSpell"
export MODEL_NAME=glyce
export FONT_TYPE=sim
export CHECKPOINT="/root/syntactic-experts/models/ECSpell/Code/Results/ecspell"
export CHECKPOINT_INDEX=300

CUDA_VISIBLE_DEVICES=0 python $DIR/Code/train_baseline.py \
	--model_name $DIR/Transformers/${MODEL_NAME} \
	--train_files $DIR/Data/traintest/final_train_ecspell.json \
	--val_files $DIR/Data/traintest/final_val_ecspell.json \
	--test_files $DIR/Data/traintest/final_val_ecspell.json \
	--cached_dir $DIR/Cache \
	--result_dir $DIR/Results \
	--glyce_config_path $DIR/Transformers/glyce_bert_both_font.json \
	--vocab_file $DIR/Data/vocab/allNoun.txt \
	--load_pretrain_checkpoint ${CHECKPOINT} \
	--checkpoint_index ${CHECKPOINT_INDEX} \
	--overwrite_cached True \
	--num_train_epochs 3 \
	--gradient_accumulation_steps 2 \
	--use_pinyin True \
	--use_word_feature False \
	--use_copy_label False \
	--compute_metrics True \
	--per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
	--save_steps 10000 \
	--logging_steps 10000 \
	--fp16 True \
	--do_test True \
