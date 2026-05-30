import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import geopandas as gpd
from shapely.geometry import shape
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
import tempfile
import os

# Damage labels and colors
LABELS = ['background', 'no-damage', 'minor', 'major', 'destroyed']
COLORS = ['#000000', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

def load_model(model_path: str, device: str = 'cpu'):
    """Load trained U-Net model from path."""
    model = smp.Unet(
        encoder_name='resnet50',
        encoder_weights=None,
        in_channels=6,
        classes=5,
        activation=None
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert PIL image to normalized numpy array."""
    img = np.array(image.convert('RGB')).astype(np.float32) / 255.0
    return img


def run_inference(model, pre_image: Image.Image, post_image: Image.Image, device: str = 'cpu') -> np.ndarray:
    """
    Run damage detection inference on before/after image pair.
    Returns damage prediction map as numpy array.
    """
    pre_img  = preprocess_image(pre_image)
    post_img = preprocess_image(post_image)

    # Resize to same size if needed
    if pre_img.shape != post_img.shape:
        post_image = post_image.resize(pre_image.size)
        post_img   = preprocess_image(post_image)

    # Stack 6 channels
    stacked = np.concatenate([pre_img, post_img], axis=2)

    # Convert to tensor
    tensor = torch.tensor(stacked)\
                  .permute(2, 0, 1)\
                  .unsqueeze(0)\
                  .float()\
                  .to(device)

    with torch.no_grad():
        output = model(tensor)
        pred   = torch.argmax(output, dim=1)\
                      .squeeze(0)\
                      .cpu()\
                      .numpy()

    return pred


def get_damage_stats(pred: np.ndarray) -> dict:
    """Calculate damage statistics from prediction map."""
    unique, counts = np.unique(pred, return_counts=True)
    total = pred.size
    stats = {}
    for u, c in zip(unique, counts):
        if u < len(LABELS):
            stats[LABELS[u]] = {
                'pixels'  : int(c),
                'percent' : round(c / total * 100, 2)
            }
    return stats


def prediction_to_colored_image(pred: np.ndarray) -> np.ndarray:
    """Convert prediction map to RGB colored image."""
    color_map = {
        0: [0,   0,   0  ],  # background - black
        1: [46,  204, 113],  # no damage  - green
        2: [241, 196, 15 ],  # minor      - yellow
        3: [230, 126, 34 ],  # major      - orange
        4: [231, 76,  60 ],  # destroyed  - red
    }
    colored = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        colored[pred == class_id] = color
    return colored


def export_geojson(pred: np.ndarray, bounds: tuple = None) -> str:
    """
    Export damage predictions as GeoJSON file.
    bounds = (left, bottom, right, top) in WGS84
    Returns path to saved GeoJSON file.
    """
    h, w = pred.shape

    # Use provided bounds or default
    if bounds:
        transform = from_bounds(*bounds, w, h)
        crs = 'EPSG:4326'
    else:
        from rasterio.transform import Affine
        transform = Affine(1, 0, 0, 0, -1, h)
        crs = None

    # Only export damaged areas
    damage_only = np.where(pred >= 2, pred, 0).astype(np.uint8)

    results = []
    for geom, val in shapes(damage_only, transform=transform):
        if val >= 2:
            results.append({
                'geometry'     : shape(geom),
                'damage_class' : int(val),
                'damage_label' : LABELS[int(val)]
            })

    if not results:
        return None

    gdf = gpd.GeoDataFrame(results)
    if crs:
        gdf.set_crs(crs, inplace=True)

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix='.geojson', delete=False)
    gdf.to_file(tmp.name, driver='GeoJSON')
    return tmp.name