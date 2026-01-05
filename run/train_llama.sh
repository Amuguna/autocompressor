nvidia-smi

# You can override the defaults by exporting environment variables before running this script.
models=(${MODELS:-"/home/work/prompt/models/Llama-3.1-8B-Instruct"})
summary_lengths=(${SUMMARY_LENGTHS:-"4 8 16"})
dataset_path=${DATASET_PATH:-"/home/work/prompt/dpc/dataset/arxiv_oai_splits_2024-05"}
block_size=${BLOCK_SIZE:-512}

total=${BATCH:-2}      # total batch size
bs=${SEQ:-2}            # batch size per device
lr=${LR:-8e-4}
warmup_steps=${WU:-5000}
save_steps=${SAVE:-5000}
num_gpus=${NUM_GPUS:-1}
segments_per_substep=${SEG:-2}
training_substeps=${SUB:-2}
num_nodes=${NUM_NODES:-1}
node=${NODE:-"localhost"}
summary_accumulation=${ACC:-true}
randomize_substeps=${RAND:-true}
segment_gradient_checkpointing=${CHECK:-false}
num_train_epochs=1

cache_dir=./.cache

export OMP_NUM_THREADS=8

for base_model in "${models[@]}"; do
    model_stub=${base_model##*/}

    for summary_length in "${summary_lengths[@]}"; do
        total_per_device=$((${total}/${num_gpus}/${num_nodes}))
        accu=$(( ${total_per_device} / ${bs} ))

        run_name_suffix="sub${training_substeps}_seg${segments_per_substep}_sum${summary_length}_lr${lr}_bsz${total}"
        if [[ ${randomize_substeps} == true ]]; then
            run_name_suffix+="_rand"
        fi
        if [[ $summary_accumulation == true ]]; then
            run_name_suffix+="_accu"
        fi
        if [[ ${segment_gradient_checkpointing} == true ]]; then
            run_name_suffix+="_check"
        fi
        run_name="ac_${model_stub}_${run_name_suffix}"

        echo "Run: ${run_name}"

        out_dir=checkpoints/$run_name
        mkdir -p $out_dir
        export WANDB_DIR=$out_dir

        header="torchrun \
--nnodes=$num_nodes \
--nproc_per_node=$num_gpus \
--rdzv-backend=c10d \
--rdzv-endpoint=$node:5546 \
train.py "

        model_url="$base_model"

        arguments=(
            --report_to wandb
            --config_name $model_url
            --tokenizer_name $model_url
            --model_name_or_path $model_url
            --gradient_accumulation_steps $accu
            --per_device_eval_batch_size $bs
            --per_device_train_batch_size $bs
            --learning_rate $lr
            --warmup_steps $warmup_steps
            --do_train
            --logging_steps 1
            --save_steps $save_steps
            --preprocessing_num_workers 6
            --dataloader_num_workers 6
            --cache_dir $cache_dir
            --dataset_name $dataset_path
            --block_size $block_size
            --add_special_tokens false
            --num_train_epochs ${num_train_epochs}
            --disable_tqdm true
            --resume_from_checkpoint true
            --log_level info
            --learning_rate $lr
            --run_name $run_name
            --output_dir $out_dir
            --summary_length $summary_length
            --accumulate_summary $summary_accumulation
            --remove_unused_columns false
            --segments_per_substep $segments_per_substep
            --training_substeps $training_substeps
            --randomize_substeps $randomize_substeps
            --segment_gradient_checkpointing $segment_gradient_checkpointing
            --bf16
            --lora
            --lora_r 16
            --lora_alpha 16
            --lora_dropout 0.05
            --lora_target_modules q_proj v_proj o_proj k_proj
            --use_fast_tokenizer false
            --lora_modules_to_save embed_summary
            $@
        )

        echo "Training ${base_model} with lr ${lr} on ${dataset_path}"
        echo "Summary length: ${summary_length}"
        echo "Outputting to $out_dir"

        echo command: echo "$header ${arguments[@]}"
        $header ${arguments[@]} 2>&1 | tee -a $out_dir/log-resume.out
    done
done
