import numpy as np
import onnxruntime as ort
import rasterio as rio
from affine import Affine

input_path = "/Users/polcorresa/Desktop/tfg/code/evolandSR/inputSRready.tif"
output_path = "/Users/polcorresa/Desktop/tfg/code/evolandSR/outputSR.tif"
model_path = "/Users/polcorresa/Desktop/tfg/code/evolandSR/sentinel2_superresolution/src/sentinel2_superresolution/models/carn_3x3x64g4sw_bootstrap.onnx"

print("Loading model...")
ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

print("Reading input image...")
with rio.open(input_path) as src:
    img = src.read()
    meta = src.meta.copy()

C, H, W = img.shape
print(f"Input shape: {img.shape}")

# Pad to 10 bands (default model expects 10 bands)
input_data = np.zeros((10, H, W), dtype=np.float32)
channels_to_copy = min(C, 10)
input_data[:channels_to_copy, :, :] = img[:channels_to_copy, :, :]

print("Running inference...")
out = ort_session.run(None, {"input": input_data[None, ...]})[0]
out = out[0] # remove batch dim

out_c, out_h, out_w = out.shape
print(f"Output shape: {out.shape}")

factor = out_h / H
print(f"Super-resolution factor: {factor}")

# Adjust affine transform
transform = meta['transform']
new_transform = Affine(transform.a / factor, transform.b, transform.c,
                       transform.d, transform.e / factor, transform.f)

meta.update({
    "count": out_c,
    "height": out_h,
    "width": out_w,
    "transform": new_transform
})

print(f"Saving to {output_path}...")
with rio.open(output_path, "w", **meta) as dst:
    dst.write(out)

print("Done!")
