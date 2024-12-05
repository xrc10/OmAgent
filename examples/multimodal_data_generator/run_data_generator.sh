mkdir -p /data23/xu_ruochen/preprocessdatawithmllm/data/omagent_data/multimodal_data_generator

# python run_cli.py \

# cd /data23/xu_ruochen/OmAgentProject/OmAgent

# python -m debugpy --listen 5679 -m examples.multimodal_data_generator.run_cli \
#     --input_file /ceph0/core_data/Om100/sft/idefics2/clevr_math.jsonl \
#     --image_folder /ceph3/core_data/dataset/ \
#     --output_file /data23/xu_ruochen/preprocessdatawithmllm/data/omagent_data/multimodal_data_generator/clevr_math.jsonl

cd /data23/xu_ruochen/OmAgentProject/OmAgent/examples/multimodal_data_generator
# python run_cli.py \
# python -m debugpy --listen 5679 run_cli.py \

python run_cli.py \
    --input_file /ceph0/core_data/Om100/sft/idefics2/clevr_math.jsonl \
    --image_folder /ceph3/core_data/dataset/ \
    --output_file /data23/xu_ruochen/preprocessdatawithmllm/data/omagent_data/multimodal_data_generator/clevr_math.jsonl