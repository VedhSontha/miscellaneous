# Camera Frame Projection Geometry

Mapping 2D pixel coordinates to 3D spatial coordinates:
- Formulation uses camera intrinsic parameters: focal length ($f_x, f_y$), principal point ($c_x, c_y$).
- Reconstructs depth mappings: $X = \frac{(u - c_x) Z}{f_x}$, $Y = \frac{(v - c_y) Z}{f_y}$.
