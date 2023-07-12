seed=9999
output_dir="./output/movierec"
base_model="decapoda-research/llama-7b-hf"
train_data="./data/movie/train.json"
val_data="./data/movie/valid.json"
instruction_model="movie"
for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in 64
        do
                mkdir -p ${output_dir}_${seed}_${sample}
                echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample outputdir: ${output_dir}_${seed}_${sample}"
                deepspeed  --num_gpus=8 finetune_rec_on_alpaca.py \
                    --base_model $base_model \
                    --train_data_path $train_data \
                    --val_data_path $val_data \
                    --output_dir $output_dir"_"$seed"_"$sample \
                    --batch_size 1280 \
                    --micro_batch_size 64 \
                    --num_epochs 400 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint $instruction_model \
                    --sample $sample \
                    --seed $1 \
                    --deepspeed ./shell/ds.json
        done
    done
done