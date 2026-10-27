"""
Generate perturbed evaluation sets for the ERIT benchmark suite.

Perturbation conditions:
  1. baseline       — original images + transcripts (copy/symlink)
  2. no_keyword     — emotion-indicative words removed from transcripts
  3. grayscale      — frames converted to grayscale
  4. blurred_faces  — Gaussian blur applied to detected face regions
  5. text_dropout   — transcript replaced with "No speech detected."
  6. image_dropout  — frame replaced with uniform gray image

Each condition produces a directory with a CSV and (for visual perturbations)
modified images.  Text-only perturbations reuse the original images via
symlinks.

Usage:
    python erit/generate_perturbations.py \
        --erit-root /scratch/vemotionsys/rmfrieske/ERIT \
        --output-dir /scratch/vemotionsys/rmfrieske/ERIT_benchmark
"""

import argparse
import csv
import os
import re
import shutil

import cv2
import numpy as np

# ── Emotion keywords to strip in no_keyword condition ──────────────────
# Covers base forms + common morphological variants
EMOTION_KEYWORDS = [
    # happiness
    r"\bhapp(?:y|iness|ily)\b",
    r"\bjoy(?:ful|ous|fully)?\b",
    r"\bdelighted?\b",
    r"\bpleased?\b",
    r"\bcheerful(?:ly)?\b",
    r"\bglad(?:ly)?\b",
    r"\bexcited?\b",
    r"\bexcitement\b",
    r"\bthrilled?\b",
    # sadness
    r"\bsad(?:ness|ly)?\b",
    r"\bunhappy\b",
    r"\bmiserable\b",
    r"\bdepressed?\b",
    r"\bsorrowful\b",
    r"\bsorrow\b",
    r"\bgrie(?:f|ving|ved)\b",
    r"\bcrying\b",
    r"\bcried\b",
    r"\btear(?:s|ful)?\b",
    # anger
    r"\bangr(?:y|ily)\b",
    r"\banger\b",
    r"\bfurious(?:ly)?\b",
    r"\bfury\b",
    r"\bmad\b",
    r"\brage\b",
    r"\birrita(?:ted|ting)\b",
    r"\bupset\b",
    r"\bannoy(?:ed|ing)?\b",
    # disgust
    r"\bdisgust(?:ed|ing)?\b",
    r"\brevolt(?:ed|ing)?\b",
    r"\bnauseat(?:ed|ing)?\b",
    r"\brepuls(?:ed|ive)\b",
    r"\bgross\b",
    r"\byuck\b",
    r"\beww?\b",
    # fear
    r"\bfear(?:ful|fully)?\b",
    r"\bafraid\b",
    r"\bscared?\b",
    r"\bscar(?:y|ier|iest)\b",
    r"\bterrif(?:ied|ying)\b",
    r"\bfrighten(?:ed|ing)?\b",
    r"\bworried?\b",
    r"\banxious(?:ly)?\b",
    r"\bnervous(?:ly)?\b",
    # surprise
    r"\bsurpris(?:ed?|ing)\b",
    r"\bshocked?\b",
    r"\bastonish(?:ed|ing)?\b",
    r"\bamazed?\b",
    r"\bamazing\b",
    r"\bstunned?\b",
    r"\bstunning\b",
    r"\bwow\b",
    r"\boh\s+my\b",
    # neutral (less common as keywords, but included)
    r"\bcalm(?:ly)?\b",
    r"\bindifferent\b",
]

KEYWORD_PATTERN = re.compile(
    "|".join(EMOTION_KEYWORDS), re.IGNORECASE
)


def remove_emotion_keywords(text):
    """Remove emotion-indicative words and collapse resulting whitespace."""
    cleaned = KEYWORD_PATTERN.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned if cleaned else "..."


def detect_face_bbox(img_bgr):
    """Detect largest face using OpenCV Haar cascade. Returns (x,y,w,h) or None."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(30, 30))
    if len(faces) == 0:
        return None
    # Return largest face by area
    areas = [w * h for (_, _, w, h) in faces]
    idx = int(np.argmax(areas))
    return tuple(faces[idx])


def blur_face_region(img_bgr, bbox, sigma=23):
    """Apply Gaussian blur to a face bounding box region."""
    x, y, w, h = bbox
    roi = img_bgr[y:y+h, x:x+w]
    ksize = sigma * 2 + 1  # must be odd
    blurred = cv2.GaussianBlur(roi, (ksize, ksize), sigma)
    out = img_bgr.copy()
    out[y:y+h, x:x+w] = blurred
    return out


def make_gray_dummy(height=224, width=224):
    """Create a uniform gray image (3-channel BGR)."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


# ── Condition generators ──────────────────────────────────────────────

def gen_baseline(rows, img_dir, out_dir):
    """Copy original images + unchanged CSV rows."""
    for row in rows:
        src = os.path.join(img_dir, row["url"])
        dst = os.path.join(out_dir, row["url"])
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    return rows


def gen_no_keyword(rows, img_dir, out_dir):
    """Remove emotion keywords from instruction text; reuse original images."""
    out_rows = []
    for row in rows:
        new_row = dict(row)
        new_row["instruction"] = remove_emotion_keywords(str(row["instruction"]))
        out_rows.append(new_row)
        # Symlink original image
        src = os.path.join(img_dir, row["url"])
        dst = os.path.join(out_dir, row["url"])
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    return out_rows


def gen_grayscale(rows, img_dir, out_dir):
    """Convert frames to 3-channel grayscale."""
    for row in rows:
        src = os.path.join(img_dir, row["url"])
        dst = os.path.join(out_dir, row["url"])
        if os.path.exists(src) and not os.path.exists(dst):
            img = cv2.imread(src)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(dst, gray3)
    return rows


def gen_blurred_faces(rows, img_dir, out_dir):
    """Apply Gaussian blur to detected face regions."""
    n_blurred = 0
    n_no_face = 0
    for row in rows:
        src = os.path.join(img_dir, row["url"])
        dst = os.path.join(out_dir, row["url"])
        if os.path.exists(src) and not os.path.exists(dst):
            img = cv2.imread(src)
            if img is not None:
                bbox = detect_face_bbox(img)
                if bbox is not None:
                    img = blur_face_region(img, bbox)
                    n_blurred += 1
                else:
                    n_no_face += 1
                cv2.imwrite(dst, img)
    print(f"    Blurred {n_blurred} faces, {n_no_face} no face detected (kept original)")
    return rows


def gen_text_dropout(rows, img_dir, out_dir):
    """Replace transcripts with placeholder; reuse original images."""
    out_rows = []
    for row in rows:
        new_row = dict(row)
        new_row["instruction"] = "No speech detected."
        out_rows.append(new_row)
        src = os.path.join(img_dir, row["url"])
        dst = os.path.join(out_dir, row["url"])
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    return out_rows


def gen_image_dropout(rows, img_dir, out_dir):
    """Replace frames with uniform gray images."""
    for row in rows:
        dst = os.path.join(out_dir, row["url"])
        if not os.path.exists(dst):
            dummy = make_gray_dummy()
            cv2.imwrite(dst, dummy)
    return rows


# ── Condition registry ────────────────────────────────────────────────

CONDITIONS = {
    "baseline":      gen_baseline,
    "no_keyword":    gen_no_keyword,
    "grayscale":     gen_grayscale,
    "blurred_faces": gen_blurred_faces,
    "text_dropout":  gen_text_dropout,
    "image_dropout": gen_image_dropout,
}


def load_all_splits(erit_root):
    """Load and merge train/dev/test CSVs with a 'split' column."""
    all_rows = []
    for split in ["train", "dev", "test"]:
        # Try ERIT_new layout first ({split}/{split}.csv),
        # then old layout (ERIT_{split}/{split}.csv)
        csv_path = os.path.join(erit_root, split, f"{split}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(erit_root, f"ERIT_{split}", f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: no CSV found for {split} in {erit_root}, skipping")
            continue
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_split"] = split
                row["_img_dir"] = os.path.dirname(csv_path)
                all_rows.append(row)
    print(f"Loaded {len(all_rows)} samples across splits")
    return all_rows


def write_csv(rows, out_path, fieldnames):
    """Write rows to CSV."""
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate perturbed evaluation sets for ERIT benchmark")
    parser.add_argument("--erit-root", required=True,
                        help="Root of ERIT splits (contains ERIT_train/, etc.)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for benchmark conditions")
    parser.add_argument("--conditions", nargs="*",
                        default=list(CONDITIONS.keys()),
                        choices=list(CONDITIONS.keys()),
                        help="Which conditions to generate (default: all)")
    args = parser.parse_args()

    all_rows = load_all_splits(args.erit_root)
    if not all_rows:
        print("ERROR: no samples loaded")
        return

    fieldnames = ["url", "instruction", "OriginalLabels", "caption",
                  "GTEmotionScore", "_split"]

    # Group by split for image directory lookup
    split_img_dirs = {}
    for row in all_rows:
        split_img_dirs[row["_split"]] = row["_img_dir"]

    for cond_name in args.conditions:
        print(f"\n{'='*50}")
        print(f"Condition: {cond_name}")
        print(f"{'='*50}")

        cond_dir = os.path.join(args.output_dir, cond_name)
        os.makedirs(cond_dir, exist_ok=True)

        gen_fn = CONDITIONS[cond_name]

        # Process per split to keep image directories correct
        all_out_rows = []
        for split in ["train", "dev", "test"]:
            split_rows = [r for r in all_rows if r["_split"] == split]
            if not split_rows:
                continue
            img_dir = split_img_dirs[split]
            print(f"  {split}: {len(split_rows)} samples (images: {img_dir})")
            out_rows = gen_fn(split_rows, img_dir, cond_dir)
            all_out_rows.extend(out_rows)

        # Save merged CSV
        csv_path = os.path.join(cond_dir, "benchmark.csv")
        write_csv(all_out_rows, csv_path, fieldnames)
        print(f"  Saved {len(all_out_rows)} rows to {csv_path}")

        # Count images
        n_imgs = len([f for f in os.listdir(cond_dir) if f.endswith(".jpg")])
        n_links = len([f for f in os.listdir(cond_dir)
                       if os.path.islink(os.path.join(cond_dir, f))])
        print(f"  Images: {n_imgs} ({n_links} symlinks)")


if __name__ == "__main__":
    main()
