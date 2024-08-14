if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PathFormer

root_path_name=./dataset/ETT/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

for pred_len in 96
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 \
      --num_nodes 7 \
      --layer_nums 3 \
      --batch_norm 0 \
      --residual_connection 1\
      --k 3\
      --d_model 4 \
      --d_ff 64 \
      --train_epochs 30\
      --patience 10\
      --lradj 'TST'\
      --itr 1 \
      --batch_size 128 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


for pred_len in 192
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 \
      --num_nodes 7 \
      --layer_nums 3 \
      --batch_norm 0 \
      --residual_connection 1\
      --k 3\
      --d_model 4 \
      --d_ff 64 \
      --train_epochs 30\
      --patience 10\
      --lradj 'TST'\
      --itr 1 \
      --batch_size 128 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

for pred_len in 336
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --patch_size_list 16 12 8 32 12 8 6 16 8 6 4 16 \
      --num_nodes 7 \
      --layer_nums 3 \
      --batch_norm 0 \
      --residual_connection 0\
      --k 3\
      --d_model 4 \
      --d_ff 64 \
      --train_epochs 30\
      --patience 10\
      --lradj 'TST'\
      --itr 1 \
      --batch_size 512 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done


for pred_len in 720
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 \
      --num_nodes 7 \
      --layer_nums 3 \
      --batch_norm 0 \
      --residual_connection 0\
      --k 2\
      --d_model 4 \
      --d_ff 64 \
      --train_epochs 30\
      --patience 10\
      --lradj 'TST'\
      --itr 1 \
      --batch_size 512 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

