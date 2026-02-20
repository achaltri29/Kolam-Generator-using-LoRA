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
    Reconstructs the image using Region-Based Smoothing:
    1. Upscale 2x.
    2. Threshold to find shapes (not just lines).
    3. Detect Circles vs Arbitrary Shapes.
    4. Smooth Boundaries.
    5. Fill shapes with original average color.
    """
    hf, wf = image.shape[:2]
    
    # 1. Upscale 2x for precision
    # INTER_CUBIC is good for upscaling
    upscaled = cv2.resize(image, (wf * 2, hf * 2), interpolation=cv2.INTER_CUBIC)
    
    if len(upscaled.shape) == 3:
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    else:
        gray = upscaled
        upscaled = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # Ensure color

    # 2. Preprocessing
    blurred = cv2.medianBlur(gray, 7) # Stronger blur to remove texture
    
    # Adaptive Threshold to separate foreground shapes from background
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 3) 
    # Note: THRESH_BINARY_INV assuming light background, dark shapes? 
    # Kolams are usually light on dark. Let's check mean.
    
    # Check if we inverted correctly. We want SHAPES to be White (255).
    # If the image is mostly black (background), and shapes are white:
    #   mean < 127. 
    #   If we used THRESH_BINARY_INV on (White on Black), result is (Black on White).
    #   We want White Shapes.
    
    # Let's trust the adaptive threshold but verify "foreground" density.
    if np.mean(binary) > 127: # If mostly white, we likely have Background=White
        binary = cv2.bitwise_not(binary)
        
    # Morphological Clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 3. Find Contours (Boundaries of shapes)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Determine Background Color
    # Sample pixels where binary is 0 (Background)
    bg_mask = (binary == 0).astype(np.uint8)
    if cv2.countNonZero(bg_mask) > 0:
        bg_color = cv2.mean(upscaled, mask=bg_mask)[:3]
    else:
        bg_color = (0, 0, 0)
        
    # Create Canvas
    canvas = np.full_like(upscaled, bg_color)
    
    # 4. Process Each Shape
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 50: # Ignore tiny noise (adjusted for 2x scale)
            continue
            
        # Calculate Circularity
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Color Sampling
        # Create a mask for just this contour to sample color
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(upscaled, mask=mask)[:3]
        
        # Draw Logic
        if circularity > 0.8:
            # It's a Circle
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            # Draw filled circle
            cv2.circle(canvas, center, radius, mean_val, -1, cv2.LINE_AA)
            
        else:
            # It's a generic shape (line segment, curve, polygon)
            # Smooth it
            epsilon = 0.002 * perimeter # Low epsilon to keep curves curved
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Draw filled smoothed shape
            cv2.drawContours(canvas, [approx], -1, mean_val, -1, cv2.LINE_AA)

    # 5. Downscale
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
                    svg_name = output_dir / f"vector_bw_{file.stem}.svg"
                    vectorize_image(enhanced_img, svg_name)
                    
        except Exception as e:
            print(f"Failed to process {file.name}: {e}")
