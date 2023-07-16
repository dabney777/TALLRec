seed=666
output_dir="./output/movierec"
base_model="decapoda-research/llama-7b-hf"
train_data="${1:-./data/movie/train.json}"
val_data="${2:-./data/movie/valid.json}"
instruction_model="./hf_ckpt"
for lr in 1e-4
do
    for dropout in 0.05
    do
        for sample in 64
        do
                mkdir -p ${output_dir}_${seed}_${sample}
                echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample outputdir: ${output_dir}_${seed}_${sample}, train_data: $train_data val_data: $val_data"
                deepspeed  finetune_rec_on_alpaca.py \
                    --base_model $base_model \
                    --train_data_path $train_data \
                    --val_data_path $val_data \
                    --output_dir $output_dir"_"$seed"_"$sample \
                    --batch_size 1280 \
                    --micro_batch_size 64 \
                    --num_epochs 120 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 16 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint $instruction_model \
                    --sample $sample \
                    --seed $seed \
                    --deepspeed ./shell/ds.json
        done
    done
done