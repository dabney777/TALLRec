CUDA_ID=$1
output_dir=$2
model_path=$(ls -d $output_dir*)
base_model="decapoda-research/llama-7b-hf"
test_data=$3
#"./data/movie/test.json"
for path in $model_path
do
    echo $path
    CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
        --base_model $base_model \
        --lora_weights $path \
        --test_data_path $test_data \
        --result_json_data $2result.json
done