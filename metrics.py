import torch
import torch.nn.functional as F

def compute_metrics(predicted, target, thresholds=[1.0, 2.0, 3.0]):
    if predicted.shape != target.shape:
        predicted = F.interpolate(predicted, size=target.shape[-2:], mode='bilinear', align_corners=False)
    mae = torch.mean(torch.abs(predicted - target)).item()
    rmse = torch.sqrt(torch.mean((predicted - target) ** 2)).item()
    # Compute Bad Pixel Error for each threshold
    bad_pixel_errors = {}
    for threshold in thresholds:
        bad_pixel_percentage = torch.mean((torch.abs(predicted - target) > threshold).float()).item() * 100
        bad_pixel_errors[f"Bad{threshold}"] = bad_pixel_percentage
    return mae, rmse, bad_pixel_errors