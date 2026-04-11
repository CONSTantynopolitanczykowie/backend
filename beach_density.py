import argparse
from pathlib import Path

import cv2
import numpy as np


def estimate_occupancy_by_area(
    image_path: str,
    sand_lower_hsv=None,
    sand_upper_hsv=None,
    roi_top_fraction: float = 0.0,
    roi_bottom_fraction: float = 1.0,
    debug_mask_output: str | None = None,
) -> dict:
  

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    max_dim = 1600
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    y1 = int(h * roi_top_fraction)
    y2 = int(h * roi_bottom_fraction)
    if y1 < 0:
        y1 = 0
    if y2 > h:
        y2 = h

    # --- maska piasku ---
    if sand_lower_hsv is None or sand_upper_hsv is None:
        # Automatyczny próg: bierzemy dolny fragment obrazu jako próbkę piasku
        sample_roi = hsv[y1:y2, :].reshape(-1, 3)
        if sample_roi.size == 0:
            sample_roi = hsv.reshape(-1, 3)


        s = sample_roi[:, 1]
        v = sample_roi[:, 2]
        sand_candidates = sample_roi[(s < 110) & (v > 80)]
        if sand_candidates.size < 500:  # jak mało kandydatów, wracamy do całej próbki
            sand_candidates = sample_roi

        h_med = float(np.median(sand_candidates[:, 0]))
        s_med = float(np.median(sand_candidates[:, 1]))
        v_med = float(np.median(sand_candidates[:, 2]))

        h_range = 12.0
        s_range = 35.0
        v_range = 45.0

        h_lower = max(0.0, h_med - h_range)
        h_upper = min(179.0, h_med + h_range)

        s_lower = max(0.0, s_med - s_range)
        s_upper = min(160.0, s_med + s_range)
        v_lower = max(0.0, v_med - v_range)
        v_upper = min(255.0, v_med + v_range)

        sand_lower = np.array([h_lower, s_lower, v_lower], dtype=np.uint8)
        sand_upper = np.array([h_upper, s_upper, v_upper], dtype=np.uint8)
    else:
        sand_lower = np.array(sand_lower_hsv, dtype=np.uint8)
        sand_upper = np.array(sand_upper_hsv, dtype=np.uint8)

    sand_mask_full = cv2.inRange(hsv, sand_lower, sand_upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sand_mask_full = cv2.morphologyEx(sand_mask_full, cv2.MORPH_OPEN, kernel, iterations=2)
    sand_mask_full = cv2.morphologyEx(sand_mask_full, cv2.MORPH_CLOSE, kernel, iterations=1)

    non_sand_full = cv2.bitwise_not(sand_mask_full)

    roi_sand = sand_mask_full[y1:y2, :]
    roi_non_sand = non_sand_full[y1:y2, :]

    total_pixels = roi_sand.size
    sand_pixels = int(np.count_nonzero(roi_sand))

    
    non_sand_pixels = total_pixels - sand_pixels

    fraction_sand = sand_pixels / total_pixels if total_pixels > 0 else 0.0
    fraction_occupied = non_sand_pixels / total_pixels if total_pixels > 0 else 0.0

    if debug_mask_output is not None:
        overlay = img.copy()

        sand_color = np.zeros_like(img)
        sand_color[:, :] = (180, 180, 180)  
        occ_color = np.zeros_like(img)
        occ_color[:, :] = (246, 178, 0)  

        roi_sand_color = sand_color[y1:y2, :]
        roi_occ_color = occ_color[y1:y2, :]

        sand_region = cv2.bitwise_and(roi_sand_color, roi_sand_color, mask=roi_sand)
        occ_region = cv2.bitwise_and(roi_occ_color, roi_occ_color, mask=roi_non_sand)

        debug = overlay.copy()
        mask_vis_roi = cv2.addWeighted(sand_region, 0.3, occ_region, 0.7, 0)
        debug[y1:y2, :] = cv2.addWeighted(overlay[y1:y2, :], 0.3, mask_vis_roi, 0.7, 0)

        text1 = f"PIASEK: {fraction_sand*100:.1f}%"
        text2 = f"ZAJETE: {fraction_occupied*100:.1f}%"
        cv2.putText(debug, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug, text2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(str(debug_mask_output), debug)

    return {
        "fraction_sand": fraction_sand,
        "fraction_occupied": fraction_occupied,
        "sand_pixels": sand_pixels,
        "non_sand_pixels": non_sand_pixels,
        "total_pixels": total_pixels,
        "scale": scale,
        "roi_y1": y1,
        "roi_y2": y2,
    }





def main():
    parser = argparse.ArgumentParser(description="Beach occupancy analysis (2 methods)")
    parser.add_argument("image", type=str, help="Path to beach photo")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save debug images",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    area_debug_path = out_dir / f"{image_path.stem}_area_mask.jpg"
    area_result = estimate_occupancy_by_area(
        str(image_path),
        debug_mask_output=str(area_debug_path),
    )

    print("=== Method 1: area-based occupancy ===")
    print(f"Sand fraction:       {area_result['fraction_sand']*100:.1f}%")
    print(f"Occupied fraction:   {area_result['fraction_occupied']*100:.1f}%")
    density_method1 = area_result["fraction_occupied"]
    print(f"Density method 1 (0-1, 1=crowded): {density_method1:.3f}")

    print("\nDebug image saved to:")
    print(f" - area mask:  {area_debug_path}")


if __name__ == "__main__":
    main()
