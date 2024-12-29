from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import *
from load_data import *
from preprocess import *
from metrics import *
from depth_estimation import *




def main():
    # Initialize paths
    """
    config_file = "C:/Me/Project fy/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    frozen_model = "C:/Me/Project fy/frozen_inference_graph.pb"
    labels_file = "C:/Me/Project fy/yolo3.txt"
    """
    dataset_root = "C:/Me/Project fy/Datasets/KITTI2015/training"


    # Load and preprocess KITTI dataset
    print("Loading KITTI dataset...")
    kitti_data = load_kitti_data(dataset_root)
    left_images = preprocess_images(kitti_data['left_images'])
    right_images = preprocess_images(kitti_data['right_images'])
    disparities = preprocess_images(kitti_data['disparity'])
    mc1, mc2, mc3, dispk = [], [], [], []
    for l, r in zip(left_images, right_images):
        a, b, c = train_rec(l, r)
        d = calculate_max_shift(l, r)
        mc1.append(a)
        mc2.append(b)
        mc3.append(c)
        dispk.append(d)
        print(f"Metrics: {a}, {b}, {c}, Max Shift: {d}")

    m = prepare_target_disparity(dispk, mc1, mc2, mc3)
    m_tensor = torch.tensor(m)
    m_tensor = m_tensor.unsqueeze(2)
    print("Shape of m:", m_tensor.shape)

    dataset = StereoDataset(left_images, right_images, disparities)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Recursive3DCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for left_img, right_img, disparity in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            left_img = left_img.to(device)
            right_img = right_img.to(device)
            disparity = disparity.to(device)
            optimizer.zero_grad()
            outputs = model(m_tensor)

            # Print shapes for debugging
            if epoch == 0 and running_loss == 0:
                print("Output shape:", outputs.shape)
                print("Target shape:", disparity.shape)

            loss = criterion(outputs, disparity)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    model.eval()
    focal_length = 645.24
    baseline = 57.07

    with torch.no_grad():
        for idx, left_image in enumerate(kitti_data['left_images'][:5]):
            #best_contour, vehicle_box = vehicle_detector.detect_vehicles(left_image)

            sample_left = torch.tensor(left_image / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            sample_right = torch.tensor(kitti_data['right_images'][idx] / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            predicted_disparity = model(m_tensor)
            predicted_disparity = torch.clamp(predicted_disparity, min=0, max=255)  # Clip values to valid range
            #target_disparity_gt = torch.tensor(disparities[idx], dtype=torch.float32).unsqueeze(0).to(device)
            target_disparity = torch.tensor(m, dtype=torch.float32).unsqueeze(0).to(device)

            print(f"Predicted min: {predicted_disparity.min()}, max: {predicted_disparity.max()}")
            print(f"Target min: {target_disparity.min()}, max: {target_disparity.max()}")

            mae, rmse, bad_pixel_errors = compute_metrics(predicted_disparity, target_disparity)
            print(f"Image Pair {idx + 1}")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            for threshold, bad_percentage in bad_pixel_errors.items():
                print(f"{threshold}: {bad_percentage:.2f}%")

            # Calculate depth map
            disparity_map = predicted_disparity.squeeze().cpu().numpy()
            depth_map = (focal_length * baseline)/(disparity_map + 1e-3)  # Adding small value to avoid division by zero
            # depth_map = depth_map.squeeze().cpu().numpy()
            # Clip depth map to reasonable values (e.g., 0 to 100 meters)
            # depth_map = np.clip(depth_map, 0, 100)
            depth_map = (focal_length * baseline) / (predicted_disparity + 1e-3)

            # Normalize for display
            disparity_display = (predicted_disparity - predicted_disparity.min()) / \
                                (predicted_disparity.max() - predicted_disparity.min())
            depth_display = np.clip(depth_map, 0, 100)


            plt.imshow(disparity_map[0], cmap='plasma')
            plt.title("Predicted Disparity Map")
            plt.colorbar()
            plt.show()
"""
            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

            # Vehicle detection visualization
            if best_contour is not None and vehicle_box is not None:
                x, y, w, h = vehicle_box
                result_image = left_image.copy()
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawContours(result_image[y:y + h, x:x + w], [best_contour], -1, (255, 0, 0), 2)
                ax1.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                ax1.set_title("Vehicle Detection")
            else:
                ax1.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
                ax1.set_title("Original Image (No Vehicle Detected)")

            # Disparity visualization
            disparity_im = ax2.imshow(disparity_map, cmap='plasma')
            plt.colorbar(disparity_im, ax=ax2, label='Disparity')
            ax2.set_title("Predicted Disparity Map")

            # Depth visualization
            depth_im = ax3.imshow(depth_map, cmap='gray')
            plt.colorbar(depth_im, ax=ax3, label='Depth (meters)')
            ax3.set_title("Calculated Depth Map")

            # Add overall title
            plt.suptitle(f"Analysis Results - Image {idx + 1}", fontsize=16)
            plt.tight_layout()
            plt.show()

            # Print depth statistics for the detected vehicle
            if best_contour is not None and vehicle_box is not None:
                x, y, w, h = vehicle_box
                vehicle_depth = depth_map[y:y + h, x:x + w]
                min_depth = np.min(vehicle_depth)
                print(f"\nVehicle {idx + 1} Depth Statistics:")
                print(f"Minimum depth: {min_depth:.2f} cm")

            if idx >= 4:
                break
"""
if __name__ == "__main__":
    main()