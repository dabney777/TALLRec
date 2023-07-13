echo $1
seed=$1
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
                deepspeed  finetune_rec.py \
                    --base_model $base_model \
                    --train_data_path $train_data \
                    --val_data_path $val_data \
                    --output_dir $output_dir"_"$seed"_"$sample \
                    --batch_size 1280 \
                    --micro_batch_size 64 \
                    --num_epochs 200 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 12 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint $instruction_model \
                    --sample $sample \
                    --seed $seed \
                    --deepspeed ./shell/ds.json
        done
    done
done
cp -r ./output /data/local/daoningjiang/Singularity/TALLRec/re_train

output_dir=./output/
model_path=$(ls -d $output_dir*)
base_model="decapoda-research/llama-7b-hf"
test_data="./data/movie/test.json"
CUDA_ID=0,1
for path in $model_path
do
    echo $path
    CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
        --base_model $base_model \
        --lora_weights $path \
        --test_data_path $test_data \
        --result_json_data $2.json
done
