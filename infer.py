import json

from PIL import Image
import src.depth_pro.depth_pro as depth_pro
from dataset.HypersimDataset import HypersimDataset
from dataset.utils import get_hdf5_array

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

dataset = HypersimDataset()
meta_json = dataset.meta_json
image_paths = []
depth_paths = []
with open(meta_json, "r", encoding="utf-8") as infile:
    for line in infile:
        entry = json.loads(line)
        image_paths.append(entry["img_path"])
        depth_paths.append(entry["depth_path"])

print(f"Total images: {len(image_paths)}, Total depths: {len(depth_paths)}")

for id in range(len(image_paths)):
    # Load and preprocess an image.
    image_path = image_paths[id]
    depth_path = depth_paths[id]
    image = get_hdf5_array(image_path)
    depth_gt = get_hdf5_array(depth_path)
    f_px = None
    image = transform(image)
    print(f"Image shape: {image.shape}")
    image = image.unsqueeze(0)
    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    print(f"prediction shape: {prediction['depth'].shape}")
    depth = prediction["depth"].unsqueeze(0)  # Depth in [m].
    predict_depth_np = depth.cpu().numpy().transpose(1, 2, 0)
    image_numpy = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    print(f"Img,prediction,gt shape: {image_numpy.shape},{predict_depth_np.shape},{depth_gt.shape}")
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.