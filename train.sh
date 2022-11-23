#!/bin/bash

# bash script is a bullshxt
source settings.sh

export CONVERTOR="repos/diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py"
export TRAINER="repos/diffusers/examples/dreambooth/train_dreambooth.py"
export BACK_CONVERTOR="repos/diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py"

py="env python"
base_model="$BASE_MODEL_PATH}/${BASE_MODEL_NAME}.ckpt"

mkdir -p "$OUTPUT_PATH"
mkdir -p "$EX_VAE_PATH"
mkdir -p "$BASE_MODEL_PATH"
mkdir -p "$DRUG_PATH"
[[ $(env conda >/dev/null 2>&1) == "127" ]] && [[ -e "$EX_VENV_PATH/bin/activate" ]] && source "$EX_VENV_PATH/bin/activate" >/dev/null 2>&1

error() {
    exit 1
}

if [[ ! -e $CONVERTOR ]]; then
    echo "Model Convert script not found. Have you run 'git submodule update --init --recursive'?"
    error
fi
    
if [[ ! -e $TRAINER ]]; then
    echo "Trainer not found. Have you run 'git submodule update --init --recursive'?"
    error
fi
    
if [[ ! -e $BACK_CONVERTOR ]]; then
    echo "Model Back Convert script not found. Have you run 'git submodule update --init --recursive'?"
    error
fi

if [[ ! -e $base_model ]];
    then if [[ ! -d $DUMP_MODEL_PATH ]]; then
        echo "Checkpoint path was not given, yank the model your like in ${BASE_MODEL_PATH}."
        error
    fi
fi

convert2diffiser() {
    $py $CONVERTOR \
        --checkpoint_path "$base_model" \
        --original_config_file "$BASE_MODEL_CONF_PATH" \
        --vae_path "$EX_VAE_PATH"  \
        --dump_path "$DUMP_MODEL_PATH" \
        --scheduler_type ddim
                    
    if [[ ! -d $DUMP_MODEL_PATH ]]; then
        echo "Cannnot convert ${base_model} into diffuser format."
        error
    fi
}

convert_back() {
    local id step
    
    while true; do
        echo -ne "(available sorted by time: $(ls -Ct $OUTPUT_PATH))\n>>> training id: "
        read -r id
        # may be unstable coz of diff among PCs
        echo -ne "(available: $([[ -d "${OUTPUT_PATH}/${id}" ]] && \
            for i in $(ls -Cd ${OUTPUT_PATH}/${id}/[0-9]* 2>/dev/nul); do \
            basename "$i" | tr "\n" " "; done))\n>>> target step: "
        read -r step
        
        diffuser_model_path="${OUTPUT_PATH}/${id:-0}/${step:-0}"
        
        if [[ ! -d $diffuser_model_path ]]; then
            echo "model not found, retry plz.."
            continue
        else
            echo "well, choosing path: ${diffuser_model_path}"
            break
        fi
    done
    
    drug_file_path="$DRUG_PATH/$(date --rfc-3339="date")/${DRUG_FILENAME_PREFIX}${step}.ckpt"
    
    $py $BACK_CONVERTOR \
            --model_path "$diffuser_model_path" \
            --checkpoint_path "$drug_file_path" \
            --unet_half
}

train() {
    if [[ ! -d $DUMP_MODEL_PATH ]]; then convert2diffiser; fi
    
    # See https://github.com/CCRcmcpe/diffusers/blob/main/examples/dreambooth/modules/args.py
    # for full parameter list
    accelerate launch $TRAINER \
        --pretrained_model_name_or_path "$DUMP_MODEL_PATH" \
        --pretrained_vae_name_or_path "${DUMP_MODEL_PATH}/vae" \
        --output_dir "$OUTPUT_PATH" \
        --config "$TRAIN_CONFIG" \
        --train_n_steps="$TO_STEPS" \
        # --train_to_epochs="$TO_EPOCH" \
        # -project "test" \
        # --run_id "test" \  
}

# entry
PS3="Show me your witchcraft: "
select option in train back all cancel
do
    echo "good. "
    case $option in
        all)
            train
            sleep 3
            convert_back
            break
            ;;
        back)
            convert_back
            break
            ;;
        train)
            train
            break
            ;;
        cancel)
            echo "cancelling..done."
            break
            ;;
    esac
done