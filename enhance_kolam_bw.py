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

def reconstruct_geometric_bw(image):
    """
    Reconstructs the image geometric shapes in Black and White:
    1. Upscale 2x.
    2. Adaptive Thresholding & Cleaning.
    3. Skeletonization (1px width).
    4. Contour Approximation (Straightens wavy lines).
    5. Draw crisp, thin, WHITE lines on BLACK canvas.
    6. Downscale.
    """
    hf, wf = image.shape[:2]
    
    # 1. Upscale 2x
    upscaled = cv2.resize(image, (wf * 2, hf * 2), interpolation=cv2.INTER_CUBIC)
    
    if len(upscaled.shape) == 3:
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    else:
        gray = upscaled

    # 2. Denoising & Smoothing
    blurred = cv2.medianBlur(gray, 5)
    
    # 3. Adaptive Thresholding
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Invert if necessary (we want white lines for skeletonize and output)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
        
    # 4. Morphological Cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 5. Skeletonize
    skeleton = skeletonize(binary // 255)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    
    # 6. Find Contours
    contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- BW HANDLING ---
    # Create black canvas
    canvas = np.zeros_like(gray)
    # -------------------
    
    # 7. Process Contours
    for cnt in contours:
        if cv2.contourArea(cnt) < 10 and cv2.arcLength(cnt, False) < 20: 
            continue
            
        epsilon = 0.003 * cv2.arcLength(cnt, False)
        approx = cv2.approxPolyDP(cnt, epsilon, False) 
        
        # Draw White Lines
        cv2.drawContours(canvas, [approx], -1, 255, 2, cv2.LINE_AA)

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

    # Ensure square resize
    h, w = img.shape[:2]
    size = min(h, w)
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    img = img[start_y:start_y+size, start_x:start_x+size]
    
    if size % 2 != 0:
        size -= 1
        img = img[:size, :size]
    
    # 2. Split into Quadrants
    mid = size // 2
    tl = img[0:mid, 0:mid]
    tr = img[0:mid, mid:size]
    bl = img[mid:size, 0:mid]
    br = img[mid:size, mid:size]

    # 3. Align Quadrants
    tr_aligned = cv2.flip(tr, 1)
    bl_aligned = cv2.flip(bl, 0)
    br_aligned = cv2.flip(br, -1)

    # 4. Select the Sharpest Quadrant
    quadrants = [tl, tr_aligned, bl_aligned, br_aligned]
    sharpness_scores = [calculate_sharpness(q) for q in quadrants]
    best_idx = np.argmax(sharpness_scores)
    best_quadrant = quadrants[best_idx]
    
    print(f"  Selected quadrant {['TL', 'TR', 'BL', 'BR'][best_idx]} (Score: {sharpness_scores[best_idx]:.2f})")

    # 5. GEOMETRIC RECONSTRUCTION (BW)
    reconstructed_quadrant = reconstruct_geometric_bw(best_quadrant)

    # 6. Reconstruct the Full Image
    new_tl = reconstructed_quadrant
    new_tr = cv2.flip(reconstructed_quadrant, 1)
    new_bl = cv2.flip(reconstructed_quadrant, 0)
    new_br = cv2.flip(reconstructed_quadrant, -1)

    top_half = np.hstack((new_tl, new_tr))
    bottom_half = np.hstack((new_bl, new_br))
    final_img = np.vstack((top_half, bottom_half))
    
    # 7. Save or Return
    if output_path:
        cv2.imwrite(str(output_path), final_img)
        print(f"Saved BW reconstructed image to {output_path}")
    
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
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    try:
        import potrace
        
        mean_val = np.mean(binary)
        if mean_val > 127: 
            bitmap_data = (binary < 127).astype(np.uint32) 
            bitmap_data = np.ascontiguousarray(bitmap_data)
        else: 
            bitmap_data = (binary > 127).astype(np.uint32)
            bitmap_data = np.ascontiguousarray(bitmap_data)
            
        bmp = potrace.Bitmap(bitmap_data)
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
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance Kolam images using symmetry.")
    parser.add_argument("--input", type=str, required=True, help="Input image file or directory")
    parser.add_argument("--output", type=str, default="enhanced_bw_output", help="Output directory")
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
                out_name = output_dir / f"enhanced_bw_{file.name}"
                cv2.imwrite(str(out_name), enhanced_img)
                
                if args.vectorize:
                    svg_name = output_dir / f"vector_bw_{file.stem}.svg"
                    vectorize_image(enhanced_img, svg_name)
                    
        except Exception as e:
            print(f"Failed to process {file.name}: {e}")
