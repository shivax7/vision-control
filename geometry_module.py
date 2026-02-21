import numpy as np

class DummyIntrinsics:
    def __init__(self):
        self.fx = 600.0
        self.fy = 600.0
        self.ppx = 320.0
        self.ppy = 240.0

def pixel_to_camera(u, v, depth_value, intrinsics, depth_scale):
    Z = depth_value * depth_scale

    if Z == 0:
        return None

    X = (u - intrinsics.ppx) * Z / intrinsics.fx
    Y = (v - intrinsics.ppy) * Z / intrinsics.fy

    return np.array([X, Y, Z])

intrinsics = DummyIntrinsics()

point = pixel_to_camera(330, 250, 1000, intrinsics, 0.001)
print("Camera 3D:", point)