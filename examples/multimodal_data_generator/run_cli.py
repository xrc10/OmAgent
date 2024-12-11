# Import required modules and components
from omagent_core.utils.container import container
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from pathlib import Path
from omagent_core.utils.registry import registry
from omagent_core.clients.devices.cli.client import DefaultClient
from omagent_core.utils.logger import logging

from agent.output_file_saver.output_file_saver import OutputFileSaver

import argparse
import json
import os


def main():
    # Initialize logging
    logging.init_logger("omagent", "omagent")

    # Set current working directory path
    CURRENT_PATH = Path(__file__).parents[0]

    # Import registered modules
    registry.import_module(project_path=CURRENT_PATH.joinpath('agent'))

    container.register_stm("RedisSTM")
    # Load container configuration from YAML file
    container.from_config(CURRENT_PATH.joinpath('container.yaml'))

    parser = argparse.ArgumentParser(description='Run the multimodal data generator')
    parser.add_argument('--input_file', type=str, default="/ceph0/core_data/Om100/sft/idefics2/clevr_math.jsonl", help='The path to the input jsonl file')
    parser.add_argument('--image_folder', type=str, default="/ceph3/core_data/dataset/", help='The path to the image folder')
    parser.add_argument('--output_file', type=str, default="/data23/xu_ruochen/preprocessdatawithmllm/data/omagent_data/multimodal_data_generator/clevr_math.jsonl", help='The path to the output jsonl file')
    args = parser.parse_args()

    print(args)

    # Initialize simple VQA workflow
    workflow = ConductorWorkflow(name='MultimodalDataGenerator')

    # Read existing ids from args.output_file
    existing_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                existing_ids.add(data['id'])

    # Initialize output file object
    # f_out = open(args.output_file, 'w')

    # Configure workflow tasks:
    # 1. Sample an output from the multimodal LLM
    task1 = simple_task(task_def_name='MultimodalLLMSampler', task_reference_name='multimodal_llm_sampler', inputs={'data': workflow.input('data')})
    # 2. Save the output to the output file
    task2 = simple_task(task_def_name='OutputFileSaver', task_reference_name='output_file_saver', inputs={'data': task1.output('data'), 'output_file': args.output_file})

    # Configure workflow execution flow: Input -> VQA
    workflow >> task1 >> task2

    # Register workflow
    workflow.register(True)

    # Initialize and start CLI client with workflow configuration
    config_path = CURRENT_PATH.joinpath('configs')
    cli_client = DefaultClient(interactor=workflow, config_path=config_path, workers=[OutputFileSaver()])

    for i, line in enumerate(open(args.input_file)):
        data = json.loads(line)
        if data['id'] in existing_ids:
            continue
        if 'id' not in data:
            data['id'] = str(i)
        data['image'] = os.path.join(args.image_folder, data['image'])
        cli_client.start_interactor_with_input(workflow_input={'data': data})

        break

    cli_client.stop_interactor_when_done()

if __name__ == "__main__":
    main()