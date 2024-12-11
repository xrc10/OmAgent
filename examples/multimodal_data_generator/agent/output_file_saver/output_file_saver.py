from pathlib import Path

from omagent_core.utils.registry import registry
from omagent_core.utils.general import read_image
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.logger import logging

CURRENT_PATH = Path(__file__).parents[0]

import json

@registry.register_worker()
class OutputFileSaver(BaseWorker):
    """
    Save the output data to the output file
    """

    def _run(self, data:dict, output_file:str, **kwargs):
        
        with open(output_file, 'a') as f_out:
            f_out.write(json.dumps(data) + '\n')
            f_out.flush() # TODO: why I have to use flush?

        return None

