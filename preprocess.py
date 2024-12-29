import numpy as np
import cv2

def preprocess_images(images, size=(512, 256)):
    return np.array([cv2.resize(img, size) for img in images])

def binarize_image(img):
    rows, cols, _ = img.shape
    binary_image = np.zeros((rows, cols, 3), dtype=np.uint8)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),         (0, 1),
                 (1, -1), (1, 0), (1, 1)]
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            for channel in range(3):
                binary_value = 0
                for k, (dx, dy) in enumerate(neighbors):
                    if img[i, j, channel] > img[i + dx, j + dy, channel]:
                        binary_value |= (1 << (7 - k))
                binary_image[i, j, channel] = binary_value
    return binary_image # Return numpy array instead of PIL Image

def rec_fld(image, patch_size):
    height, width, _ = image.shape
    binary_rec = []
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = image[y:y + patch_size, x:x + patch_size]
            binary_rec.append(binarize_image(patch))
    return np.array(binary_rec)

def calculate_max_shift(left_img, right_img, max_disparity=192):
    m_arr = []
    for d in range(1, max_disparity):
        shifted_right_img = np.roll(right_img, shift=-d, axis=1)
        m_arr.append(np.mean(np.abs(left_img[:, d:] - shifted_right_img[:, d:])))
    if not m_arr: #handle the case where m_arr might be empty
      return 0
    return np.argmin(m_arr) + 1

def train_rec(l_img, r_img):
    max_shift = calculate_max_shift(l_img, r_img)
    patches = [55, 25, 15]
    p_values = []
    for p in patches:
        l_patch = rec_fld(l_img, patch_size=p)
        r_patch = rec_fld(r_img, patch_size=p)
        rectified_r_patch = np.zeros_like(r_patch)
        for i in range(r_patch.shape[0]):
            shift = min(max_shift, r_patch.shape[2]-1) #Corrected this line, important!
            rectified_r_patch[i] = np.roll(r_patch[i], shift=-shift, axis=1)
        xor = np.bitwise_xor(l_patch, rectified_r_patch)
        p_values.append(np.sum(xor))
    return tuple(p_values)

def prepare_target_disparity(disp_k, mc1, mc2, mc3):
    m = []  # To store the (3, 2, 256, 512) array

    for i in range(len(disp_k)):
        # Normalize disp_k for the current image
        normalized_disp_k = disp_k[i] / max(disp_k) if max(disp_k) != 0 else 0
        disp_k_map = np.full((256, 512), normalized_disp_k, dtype=np.float32)  # Shape: (256, 512)

        # Compute the combined metric for the current image
        combined_metric = (
            (1 - mc1[i] / max(mc1) if max(mc1) != 0 else 0) / 3 +
            (1 - mc2[i] / max(mc2) if max(mc2) != 0 else 0) / 3 +
            (1 - mc3[i] / max(mc3) if max(mc3) != 0 else 0) / 3
        )
        metric_map = np.full((256, 512), combined_metric, dtype=np.float32)  # Shape: (256, 512)

        # Stack disp_k and metric maps along the channel dimension
        combined_map = np.stack([disp_k_map, metric_map], axis=0)  # Shape: (2, 256, 512)
        m.append(combined_map)

    # Convert to numpy array with shape (3, 2, 256, 512)
    m = np.array(m, dtype=np.float32)

    return m
