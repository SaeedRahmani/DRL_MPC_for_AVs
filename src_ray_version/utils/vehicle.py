import numpy as np

class Vehicle:
    
    LENGTH = 2.5 # wheelbase
    LENGTH_REAR = LENGTH / 2 # Distance from rear axle to center of mass
        
    def __init__(
        self,
        index,
        position,
        vectorized_speed,
        heading,
        sinh: float,
        cosh: float,
    ):
        self.index = index
        self.is_ego = True if index == 0 else False
        
        self.position = position
        self.vectorized_speed = vectorized_speed
        self.heading = heading
        self.speed = np.linalg.norm(self.vectorized_speed)
        self.sinh = sinh
        self.cosh = cosh
        self.max_acceleration = 3.5
        self.max_deceleration = -10