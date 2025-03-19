from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from typing import Optional

class ChannelFactoryManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChannelFactoryManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, domain_id: int, network_interface_name: Optional[str]) -> None:
        """
        Initialize the ChannelFactory if it hasn't been initialized yet.
        
        Args:
            domain_id: The domain ID for initialization
            network_interface_name: The network interface name
        """
        if not cls._initialized:
            ChannelFactoryInitialize(domain_id, network_interface_name)
            cls._initialized = True

    @classmethod
    def is_initialized(cls) -> bool:
        """
        Check if ChannelFactory has been initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return cls._initialized 