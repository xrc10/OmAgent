# Import required modules and components
import os

os.environ["OMAGENT_MODE"] = "pro"
from pathlib import Path

from omagent_core.clients.devices.webpage import WebpageClient
from omagent_core.utils.container import container
from omagent_core.utils.logger import logging
from omagent_core.utils.registry import registry
from agent.workflow import construct_workflow

logging.init_logger("omagent", "omagent", level="INFO")

# Set current working directory path
CURRENT_PATH = root_path = Path(__file__).parents[0]

# Import registered modules
registry.import_module(project_path=CURRENT_PATH.joinpath("agent"))

container.register_stm("RedisSTM")
# Load container configuration from YAML file
# container.from_config(CURRENT_PATH.joinpath("container.yaml"))
container.from_config("/home/unitree/zl/proj/OmAgent/examples/ut_dog/container.yaml")

# Initialize and start app client with workflow configuration
agent_client = WebpageClient(interactor=construct_workflow(), config_path=CURRENT_PATH.joinpath("configs"))
agent_client.start_interactor()
