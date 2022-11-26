#!/bin/bash

# bash script is a bullshxt
source settings.sh

export CONVERTOR="repos/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py"
export TRAINER="repos/diffusers/examples/dreambooth/train_dreambooth.py"
export BACK_CONVERTOR="repos/diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py"

py="env python"
base_model="${BASE_MODEL_PATH}/${BASE_MODEL_NAME}.ckpt"

mkdir -p "$OUTPUT_PATH"
mkdir -p "$BASE_MODEL_PATH"
mkdir -p "$EX_VAE_PATH"
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
    cat <<'EOF'
###################################
~~~~    Molding your wand...   ~~~~
###################################
EOF

# scheduler_type: pndm', 'lms', 'ddim


    $py $CONVERTOR \
        --checkpoint_path "$base_model" \
        --original_config_file "$BASE_MODEL_CONF_PATH" \
        --vae_path "${EX_VAE_PATH}/${EX_VAE_FILE}"  \
        --dump_path "$DUMP_MODEL_PATH" \
        --scheduler_type ddim 
        # --extract_ema \
        # --half-unet
                    
    if [[ ! -d $DUMP_MODEL_PATH ]]; then
        echo "Cannnot convert ${base_model} into diffuser format."
        error
    fi
}

convert_back() {
    local id step
    
    cat <<'EOF'
#####################################
~~~~    Drugs comeing soon...    ~~~~
#####################################
EOF

    local ids="$(ls -Ct $OUTPUT_PATH)"
        
    echo -ne "(available sorted by time: $ids)\n>>> training id (empty to the latest): "
    read -r id  
    # set the newest default
    [[ -z $id ]] && id=${ids[0]}
        
    # may be unstable coz of diff among PCs
    local steps="$([[ -d "${OUTPUT_PATH}/${id}" ]] && \
        for i in $(ls -Cd ${OUTPUT_PATH}/${id}/[0-9]* 2>/dev/null); do basename "$i" | tr "\n" " "; done)"
        
    echo -ne "(available: $steps)\n>>> target step (exp: 200 or 100 200): "
    read -ra steps
        
    for step in "${steps[@]}"; do
        diffuser_model_path="${OUTPUT_PATH}/${id:-0}/${step:-0}"
        
        if [[ ! -d $diffuser_model_path ]]; then
            echo "path $diffuser_model_path not found"
            continue
        else
            echo "well, choosing path: ${diffuser_model_path}"
        fi
    
    drug_file_path="$DRUG_PATH/$(date --rfc-3339="date")"
    mkdir -p $drug_file_path
    
    $py $BACK_CONVERTOR \
            --model_path "$diffuser_model_path" \
            --checkpoint_path "$drug_file_path/${DRUG_FILENAME_PREFIX}${step}.ckpt"
            # --text_encoder
            # --text_encoder_dtype $PAK_TTE_DTYPE
            # --vae
            # --vae_dtype $PAK_VAE_DTYPE
            # --unet_dtype $PAK_UNET_DTYPE
            # --unet_half
    echo "Coverted.."
    done
}

train() {
    if [[ ! -d $DUMP_MODEL_PATH ]]; then convert2diffiser; else echo "Found baseline: $DUMP_MODEL_PATH"; fi
    
    # See https://github.com/CCRcmcpe/diffusers/blob/main/examples/dreambooth/modules/args.py
    # for full parameter list
    
    cat <<'EOF'
###############################
~~~~     Enchanting...     ~~~~
###############################
EOF
    
    accelerate launch $TRAINER \
        --pretrained_model_name_or_path "$DUMP_MODEL_PATH" \
        --pretrained_vae_name_or_path "${DUMP_MODEL_PATH}/vae" \
        --output_dir "$OUTPUT_PATH" \
        --config "$TRAIN_CONFIG" \
        --train_n_steps="$TO_STEPS" 
        # --train_to_epochs="$TO_EPOCH" \
        # -project "test" \
        # --run_id "test" \  
}

clean() {
    cat <<'EOF'
###########################
~~~~     Wiping...     ~~~~
###########################
EOF
    echo -ne "WARNING: all relevant file will be remove, you know about your risk?\n[y or other]: "
    read -r yn
    if [[ "$yn" != "y" ]]; then 
        echo "Wiping cancelled..."
        exit 0; 
    fi
    
    local path_list=($OUTPUT_PATH)
    for i in ${path_list[@]}; do
        rm -rf "$i"
    done
    
    echo "Current autodl-tmp usage: $(du -sh ${AUTODL_TMP})"
    echo "Wow, such an empty.."
}

# entry
PS3="Show me your witchcraft: "
select option in train back all clean cancel
do
    echo "Good. "
    case $option in
        all)
            train
            sleep 10
            convert_back
            break ;;
        back)
            convert_back
            break ;;
        train)
            train
            break ;;
        clean)
            clean
            break ;;
        cancel)
            echo "cancelling..done."
            break ;;
    esac
done