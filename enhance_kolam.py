import cv2
import numpy as np
import os
import argparse
from pathlib import Path

from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev

def calculate_sharpness(image):
    """
    Calculate the sharpness of an image using the Variance of Laplacian method.
    Higher variance -> Sharper edges.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def reconstruct_geometric(image):
    """
    Reconstructs the image geometric shapes with high fidelity AND color:
    1. Upscale 2x for precision.
    2. Adaptive Thresholding & Cleaning.
    3. Skeletonization (1px width).
    4. Contour Approximation (Straightens wavy lines).
    5. Color Sampling (Restores original colors).
    6. Draw crisp, thin, colored lines.
    7. Downscale.
    """
    hf, wf = image.shape[:2]
    
    # 1. Upscale 2x
    # INTER_CUBIC is good for upscaling
    upscaled = cv2.resize(image, (wf * 2, hf * 2), interpolation=cv2.INTER_CUBIC)
    
    if len(upscaled.shape) == 3:
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    else:
        gray = upscaled
        # Ensure upscaled is 3-channel for color sampling later
        upscaled = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2. Denoising & Smoothing
    # Median blur is great for "salt and pepper" noise and preserving edges
    blurred = cv2.medianBlur(gray, 5)
    
    # 3. Adaptive Thresholding
    # Better than global Otsu for varying lighting/gradients
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Invert if necessary (we want white lines for skeletonize)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
        
    # 4. Morphological Cleaning
    # Close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 5. Skeletonize
    skeleton = skeletonize(binary // 255)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    
    # 6. Find Contours
    contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- COLOR HANDLING ---
    # Detect Background Color (median of pixels where binary mask is 0)
    bg_mask = (binary == 0).astype(np.uint8)
    # Check if we have enough background pixels
    if cv2.countNonZero(bg_mask) > 0:
        # cv2.mean returns scalar for gray or tuple for color. We want the color.
        # But mean might be "muddy". Median is safer but slower. 
        # Let's stick to mean for background as it's usually large area.
        bg_color = cv2.mean(upscaled, mask=bg_mask)[:3]
    else:
        bg_color = (0, 0, 0) # Fallback black
        
    # Create canvas filled with background color
    canvas = np.full_like(upscaled, bg_color)
    # ----------------------
    
    # 7. Process Contours
    for cnt in contours:
        # Filter small noise
        if cv2.contourArea(cnt) < 10 and cv2.arcLength(cnt, False) < 20: 
            continue
            
        # Straighten Wobbly Lines
        epsilon = 0.003 * cv2.arcLength(cnt, False)
        approx = cv2.approxPolyDP(cnt, epsilon, False) 
        
        # --- COLOR SAMPLING ---
        # Sample color from the specific segment
        # cnt contains all points along the skeleton. We can sample their colors.
        pts = cnt.reshape(-1, 2)
        # Clip to safe bounds
        pts[:, 0] = np.clip(pts[:, 0], 0, upscaled.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, upscaled.shape[0] - 1)
        
        # Extract pixel values: upscaled[y, x]
        colors = upscaled[pts[:, 1], pts[:, 0]]
        
        # Use Median color of the line segment to avoid noise
        if len(colors) > 0:
            segment_color = np.median(colors, axis=0).astype(int).tolist()
        else:
            segment_color = (255, 255, 255) # Fallback white
        # ----------------------
        
        # Draw Lines
        cv2.drawContours(canvas, [approx], -1, segment_color, 2, cv2.LINE_AA)

    # 8. Downscale back to original size
    final_reconstructed = cv2.resize(canvas, (wf, hf), interpolation=cv2.INTER_AREA)
    
    return final_reconstructed

def enforce_symmetry(image_path, output_path=None):
    """
    Enforces 4-fold rotational symmetry by selecting the sharpest quadrant,
    reconstructing its geometry (smoothing), and mirroring it.
    """
    # 1. Load Image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # Ensure square resize if necessary (most Kolams are square)
    h, w = img.shape[:2]
    size = min(h, w)
    # Center crop to square
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    img = img[start_y:start_y+size, start_x:start_x+size]
    
    # Ideally resize to even dimensions for perfect splitting
    if size % 2 != 0:
        size -= 1
        img = img[:size, :size]
    
    # 2. Split into Quadrants
    mid = size // 2
    # TL: Top-Left
    tl = img[0:mid, 0:mid]
    # TR: Top-Right
    tr = img[0:mid, mid:size]
    # BL: Bottom-Left
    bl = img[mid:size, 0:mid]
    # BR: Bottom-Right
    br = img[mid:size, mid:size]

    # 3. Align Quadrants to Top-Left Orientation
    # TR needs horizontal flip to match TL
    tr_aligned = cv2.flip(tr, 1)
    # BL needs vertical flip to match TL
    bl_aligned = cv2.flip(bl, 0)
    # BR needs both horizontal and vertical flip to match TL
    br_aligned = cv2.flip(br, -1)

    # 4. Select the Sharpest Quadrant
    quadrants = [tl, tr_aligned, bl_aligned, br_aligned]
    sharpness_scores = [calculate_sharpness(q) for q in quadrants]
    best_idx = np.argmax(sharpness_scores)
    best_quadrant = quadrants[best_idx]
    
    print(f"  Selected quadrant {['TL', 'TR', 'BL', 'BR'][best_idx]} (Score: {sharpness_scores[best_idx]:.2f})")

    # 5. GEOMETRIC RECONSTRUCTION on the Best Quadrant
    # This fixes wobbly lines and curves BEFORE mirroring
    reconstructed_quadrant = reconstruct_geometric(best_quadrant)

    # 6. Reconstruct the Full Image
    new_tl = reconstructed_quadrant
    new_tr = cv2.flip(reconstructed_quadrant, 1)
    new_bl = cv2.flip(reconstructed_quadrant, 0)
    new_br = cv2.flip(reconstructed_quadrant, -1)

    # Stitch
    top_half = np.hstack((new_tl, new_tr))
    bottom_half = np.hstack((new_bl, new_br))
    final_img = np.vstack((top_half, bottom_half))
    
    # 7. Save or Return
    if output_path:
        cv2.imwrite(str(output_path), final_img)
        print(f"Saved geometrically reconstructed image to {output_path}")
    
    return final_img

def vectorize_image(image, output_svg_path):
    """
    Vectorizes a raster image using Potrace via pypotrace.
    Requires 'potrace' library to be installed.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Thresholding to binary
    # Use Otsu's thresholding for adaptive binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    try:
        import potrace
        
        # Determine if we should trace black or white. 
        # Potrace traces "foreground" (1s) on "background" (0s).
        # Assuming we want to trace the pattern lines.
        # If the image is mostly white, the pattern is likely black lines.
        mean_val = np.mean(binary)
        if mean_val > 127: # Light background, dark lines
            # Invert so lines become white (1) and background black (0) for Potrace logic?
            # Actually pypotrace Bitmap takes a buffer or numpy array.
            # If we pass data, 0 is "off", non-zero is "on".
            # So to trace lines, lines must be non-zero.
            # If current lines are black (0), we need to invert.
            bitmap_data = (binary < 127).astype(np.uint32) 
            # Note: pypotrace might expect contiguous array.
            bitmap_data = np.ascontiguousarray(bitmap_data)
        else: # Dark background, light lines
            bitmap_data = (binary > 127).astype(np.uint32)
            bitmap_data = np.ascontiguousarray(bitmap_data)
            
        # Create bitmap from numpy array
        # Check if potrace.Bitmap accepts numpy array directly (it usually does via buffer interface)
        bmp = potrace.Bitmap(bitmap_data)
        
        # Optimizations: turdsize (despeckle), alphamax (corner threshold)
        path = bmp.trace(turdsize=2, alphamax=1.0)
        
        with open(output_svg_path, "w") as f:
            f.write(f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="{image.shape[1]}" height="{image.shape[0]}">\n')
            f.write(f'<path d="')
            for curve in path:
                start = curve.start_point
                f.write(f"M {start.x},{start.y} ")
                for segment in curve:
                    if segment.is_corner:
                        c = segment.c
                        end = segment.end_point
                        f.write(f"L {c.x},{c.y} L {end.x},{end.y} ")
                    else:
                        c1 = segment.c1
                        c2 = segment.c2
                        end = segment.end_point
                        f.write(f"C {c1.x},{c1.y} {c2.x},{c2.y} {end.x},{end.y} ")
                f.write("Z ") # Close path
            f.write(f'" fill="black" stroke="none" />\n')
            f.write("</svg>")
            
        print(f"Saved SVG to {output_svg_path}")
        return True

    except ImportError:
        print("Warning: 'potrace' module import failed. Install pypotrace.")
        return False
    except Exception as e:
        print(f"Error during vectorization: {e}")
        # Print more detail for debugging
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance Kolam images using symmetry.")
    parser.add_argument("--input", type=str, required=True, help="Input image file or directory")
    parser.add_argument("--output", type=str, default="enhanced_output", help="Output directory")
    parser.add_argument("--vectorize", action="store_true", help="Attempt to vectorize the output")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
    else:
        print("Invalid input path")
        exit(1)
        
    for file in files:
        print(f"Processing {file.name}...")
        try:
            enhanced_img = enforce_symmetry(file)
            if enhanced_img is not None:
                out_name = output_dir / f"enhanced_{file.name}"
                cv2.imwrite(str(out_name), enhanced_img)
                
                if args.vectorize:
                    svg_name = output_dir / f"vector_{file.stem}.svg"
                    vectorize_image(enhanced_img, svg_name)
                    
        except Exception as e:
            print(f"Failed to process {file.name}: {e}")
