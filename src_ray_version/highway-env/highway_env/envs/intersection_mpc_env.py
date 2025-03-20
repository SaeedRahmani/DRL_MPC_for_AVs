import highway_env
from highway_env.envs import IntersectionEnv


# highway_env._register_highway_envs()


class IntersectionMpcEnv(IntersectionEnv):
    """ An intersection environment with MPC solver inside. """
    def __init__(
        self,
    ):
        super().__init__()
    
    
        
    def __str__(self) -> str:
        return f"IntersectionMpcEnv()"
    
    def __repr__(self) -> str:
        return self.__str__()
    