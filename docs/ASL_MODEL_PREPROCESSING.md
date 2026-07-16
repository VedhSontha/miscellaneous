# ASL Image Preprocessing Steps

Data formatting pipeline for Sign Language classification:
- Resizes inputs to $224\times224$ pixels.
- Normalizes pixel channels using ImageNet mean/standard deviation.
- Converts training inputs to tensor arrays.
