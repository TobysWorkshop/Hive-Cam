import numpy as np
import json

def compute_polygon_area(quad):
    """Compute the area of a quadrilateral using the shoelace formula."""
    n = len(quad)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += quad[i][0] * quad[j][1]
        area -= quad[j][0] * quad[i][1]
    return abs(area) / 2.0

def compute_average_polygon(entrance_json_path):
    # Load entrance polygons from JSON
    try:
        with open(entrance_json_path, "r") as f:
            all_entrance_polygons = json.load(f)
    except Exception as e:
        print(f"[Error] Could not load entrance polygons from {entrance_json_path}: {e}")
        return None
    
    # List to hold valid polygons and their areas
    valid_polygons = []
    polygon_areas = []
    
    # Find the first valid polygon as reference
    ref_poly = None
    first_frame_key = None
    for frame_key in sorted(all_entrance_polygons.keys(), key=int):
        poly = all_entrance_polygons.get(frame_key, [])
        if not poly or not isinstance(poly, list) or len(poly) == 0:
            #print(f"Warning: No valid polygon data for frame {frame_key}. Skipping.")
            continue
        poly = poly[0]  # Extract inner list
        if not isinstance(poly, list) or len(poly) != 4 or not all(
            isinstance(p, list) and len(p) == 2 and all(isinstance(c, (int, float)) for c in p) for p in poly
        ):
            #print(f"Warning: Invalid polygon for frame {frame_key}: expected 4 points with [x, y] coordinates, got {poly}. Skipping.")
            continue
        ref_poly = np.array(poly, dtype=np.float32)
        first_frame_key = frame_key
        break
    
    if ref_poly is None:
        print("[Error] No valid polygons found")
        return None
    
    # Compute reference centroid
    ref_centroid = np.mean(ref_poly, axis=0)
    valid_polygons.append(ref_poly)
    polygon_areas.append(compute_polygon_area(ref_poly))
    
    # Process other polygons
    for frame_key in sorted(all_entrance_polygons.keys(), key=int):
        if frame_key == first_frame_key:
            continue  # Skip reference
        poly = all_entrance_polygons.get(frame_key, [])
        if not poly or not isinstance(poly, list) or len(poly) == 0:
            #print(f"Warning: No valid polygon data for frame {frame_key}. Skipping.")
            continue
        poly = poly[0]  # Extract inner list
        if not isinstance(poly, list) or len(poly) != 4 or not all(
            isinstance(p, list) and len(p) == 2 and all(isinstance(c, (int, float)) for c in p) for p in poly
        ):
            #print(f"Warning: Invalid polygon for frame {frame_key}: expected 4 points with [x, y] coordinates, got {poly}. Skipping.")
            continue
        poly_array = np.array(poly, dtype=np.float32)
        area = compute_polygon_area(poly_array)
        polygon_areas.append(area)
        valid_polygons.append(poly_array)
    
    if len(valid_polygons) == 0:
        print("[Error] No valid polygons found")
        return None
    
    # Filter polygons by area (exclude those < 50% of median area)
    median_area = np.median(polygon_areas)
    area_threshold = 0.5 * median_area
    filtered_polygons = [
        poly for poly, area in zip(valid_polygons, polygon_areas)
        if area >= area_threshold
    ]
    
    # Align polygons to reference by translating centroids
    aligned_polygons = []
    for poly in filtered_polygons:
        centroid = np.mean(poly, axis=0)
        translation = ref_centroid - centroid
        aligned_poly = poly + translation
        aligned_polygons.append(aligned_poly)
    
    # Average the aligned polygons
    average_quad = np.mean(aligned_polygons, axis=0)
    
    # Output as list of lists
    average_polygon = average_quad.tolist()
    
    return average_polygon


#if __name__ == "__main__":
    #entrance_json_path = "data/yolo_detections/polygons.json"
    #average_polygon = compute_average_polygon(entrance_json_path)
    #if average_polygon is not None:
        #print(average_polygon)