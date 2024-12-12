CUDA_VISIBLE_DEVICES=4 \
    python3.10 /home/hhs/main/llama3/chr/Plan_Solve/2step/2Step.py \
        --input-path /home/hhs/main/llama3/chr/Data/original_input.json \
        --example-path /home/hhs/main/llama3/chr/Plan_Solve/2step/prompt4inference_2Step_ps_4shot.txt \
        --output-path ./esnli_2step_Plan_Solve.json\