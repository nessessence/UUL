import torch
import shutil
from pathlib import Path

# ------------------------------------------------------------------
# 1.  Load the 500 validation-set IDs you saved earlier
# ------------------------------------------------------------------
n_sample = 5000

paired_prompt = torch.load("data_root/data/real_data/coco/id_caption_coco30k_seed123.pth")
val_ids = [id_ for id_,caption_ in paired_prompt[:n_sample]]
        
        


# val_ids_500: list[int] = data["coco_ids"]      # list of 500 ints
# ------------------------------------------------------------------
# 2.  Where is your extracted COCO *validation* set?
#     Point this to the folder that contains files like
#     “COCO_val2014_000000000042.jpg”.
# ------------------------------------------------------------------
val_root = Path("data_root/data/real_data/coco/val2014")            # <<< change me

# ------------------------------------------------------------------
# 3.  Destination for the 500 images
# ------------------------------------------------------------------
dest_root = Path("data_root/data/real_data/coco/coco30k_5000")
dest_root.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 4.  Copy each image whose ID exists in val2014
# ------------------------------------------------------------------
missing = []

for img_id in val_ids:
    filename = f"COCO_val2014_{img_id:012d}.jpg"   # zero-pad to 12 digits
    src = val_root / filename
    if src.is_file():
        shutil.copy2(src, dest_root / filename)    # copy2 → keep timestamps
    else:
        missing.append(img_id)

print(f"Copied {len(val_ids) - len(missing)} / {len(val_ids)} images "
      f"to {dest_root.resolve()}")

if missing:
    print("⚠️  These IDs were not found in val2014:")
    print(missing)
