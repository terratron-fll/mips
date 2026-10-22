import os
import sys
from typing import Dict, Any
from PIL import Image

def seed_cat_dataset(
    target_dir: str = "data/raw/intro",
    count_per_class: int = 30,
    max_size: int = 448,
) -> Dict[str, Any]:
    """Streams and downsamples cat and not_cat (dog) images from Hugging Face Datasets

    into the intro raw data folders.
    """
    cat_dir = os.path.join(target_dir, "cat")
    not_cat_dir = os.path.join(target_dir, "not_cat")
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(not_cat_dir, exist_ok=True)

    cat_count = 0
    not_cat_count = 0
    source_name = "Hugging Face (Bingsu/Cat_and_Dog)"

    try:
        from datasets import load_dataset

        ds = load_dataset("Bingsu/Cat_and_Dog", split="train", streaming=True)
        # In Bingsu/Cat_and_Dog: label 0 is Cat, label 1 is Dog (not_cat)
        for item in ds:
            if cat_count >= count_per_class and not_cat_count >= count_per_class:
                break

            label = item.get("labels")
            pil_img = item.get("image")
            if pil_img is None:
                continue

            pil_img = pil_img.convert("RGB")
            pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            if label == 0 and cat_count < count_per_class:
                dest = os.path.join(cat_dir, f"hf_cat_{cat_count:03d}.jpg")
                pil_img.save(dest, "JPEG", quality=85, optimize=True)
                cat_count += 1
            elif label == 1 and not_cat_count < count_per_class:
                dest = os.path.join(not_cat_dir, f"hf_not_cat_{not_cat_count:03d}.jpg")
                pil_img.save(dest, "JPEG", quality=85, optimize=True)
                not_cat_count += 1

    except Exception as e:
        source_name = f"Synthetic Fallback (Network error: {str(e)})"
        # Provide fallback starter images for offline testing
        while cat_count < count_per_class:
            img = Image.new("RGB", (300, 300), color=(200, 100, 100 + cat_count * 5))
            dest = os.path.join(cat_dir, f"cat_{cat_count:03d}.jpg")
            img.save(dest, "JPEG")
            cat_count += 1

        while not_cat_count < count_per_class:
            img = Image.new("RGB", (300, 300), color=(50, 150, 50 + not_cat_count * 5))
            dest = os.path.join(not_cat_dir, f"not_cat_{not_cat_count:03d}.jpg")
            img.save(dest, "JPEG")
            not_cat_count += 1

    return {
        "cat_saved": cat_count,
        "not_cat_saved": not_cat_count,
        "source": source_name,
        "target_dir": target_dir,
    }

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/intro"
    res = seed_cat_dataset(target_dir=out_dir, count_per_class=30)
    print("Seed complete:", res)
