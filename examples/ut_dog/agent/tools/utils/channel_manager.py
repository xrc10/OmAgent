from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from typing import Optional
import time

class ChannelFactoryManager:
    _instance = None
    _initialized = False
    _sport_client = None

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
            msc = MotionSwitcherClient()
            msc.SetTimeout(5.0)
            msc.Init()
            
            # Initialize sport client as a class attribute
            cls._sport_client = SportClient()
            cls._sport_client.SetTimeout(5.0)
            cls._sport_client.Init()
            
            code, data = msc.CheckMode()
            if code == 0 and data["name"] == "normal":
                pass
            else:
                msc.SelectMode("normal")
                code = -1
                while code != 0:
                    code = cls._sport_client.Trigger()
                    time.sleep(1)
            cls._initialized = True

    @classmethod
    def is_initialized(cls) -> bool:
        """
        Check if ChannelFactory has been initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return cls._initialized

    @classmethod
    def get_sport_client(cls) -> SportClient:
        """
        Get the singleton instance of SportClient.
        
        Returns:
            SportClient: The initialized SportClient instance
        
        Raises:
            RuntimeError: If ChannelFactoryManager is not initialized
        """
        if not cls._initialized:
            raise RuntimeError("ChannelFactoryManager must be initialized first")
        return cls._sport_client 
