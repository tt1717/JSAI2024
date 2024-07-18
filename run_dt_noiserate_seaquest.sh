# Decision Transformer (DT)
# train_typeとeval_typeを適宜変更して実行する 
# Seaquest
for noise_rate in 0.0 0.2 0.4 0.6 0.8 1.0
do
    python run_dt_atari.py  -w True --train_type "spe" --noise_rate $noise_rate --eval_type "clean" --data_dir_prefix "./clean/" --seed 123 --context_length 30 --epochs 5 --model_type 'dt' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
done
# Seaquest
for noise_rate in 0.0 0.2 0.4 0.6 0.8 1.0
do
    python run_dt_atari.py  -w True --train_type "spe" --noise_rate $noise_rate --eval_type "gaus" --data_dir_prefix "./clean/" --seed 123 --context_length 30 --epochs 5 --model_type 'dt' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
done
# Seaquest
for noise_rate in 0.0 0.2 0.4 0.6 0.8 1.0
do
    python run_dt_atari.py  -w True --train_type "spe" --noise_rate $noise_rate --eval_type "shot" --data_dir_prefix "./clean/" --seed 123 --context_length 30 --epochs 5 --model_type 'dt' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
done
# Seaquest
for noise_rate in 0.0 0.2 0.4 0.6 0.8 1.0
do
    python run_dt_atari.py  -w True --train_type "spe" --noise_rate $noise_rate --eval_type "imp" --data_dir_prefix "./clean/" --seed 123 --context_length 30 --epochs 5 --model_type 'dt' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
done
