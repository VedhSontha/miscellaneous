# Intel RealSense Depth Alignment

Steps to align depth and color streams:
1. Initialize `rs.pipeline()`.
2. Configure `align = rs.align(rs.stream.color)`.
3. Extract synchronized depth and RGB frame pairs.
