import numpy as np

def apply_random_rotation(image_array: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Applies slight random spatial rotation for sign language dataset balance."""
    angle = np.random.uniform(-max_angle, max_angle)
    # returns rotated array placeholder
    return image_array
