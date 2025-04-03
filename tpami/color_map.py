import cv2
import numpy as np
import sys


gray_image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
raw_image = cv2.imread(sys.argv[2])

# Normalize the grayscale image to be within [0, 255]
heatmap_min, heatmap_max = np.min(gray_image), np.max(gray_image)
norm_heatmap = 255.0 * (gray_image - heatmap_min) / (heatmap_max - heatmap_min)

# Apply a colormap to make the heatmap visualizable
color_heatmap = cv2.applyColorMap(norm_heatmap.astype(np.uint8), cv2.COLORMAP_JET)

# Resize the color heatmap to match the original image
color_heatmap_resized = cv2.resize(color_heatmap, (raw_image.shape[1], raw_image.shape[0]))

# Blend the original image and the heatmap
factor = 0.5  # Weight of the original image in the blend
blended_image = cv2.addWeighted(raw_image, factor, color_heatmap_resized, 1 - factor, 0)

cv2.imwrite(sys.argv[3], blended_image)