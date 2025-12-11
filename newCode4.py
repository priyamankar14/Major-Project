import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----------------------------
# 🔧 CONFIGURATION — UPDATED: Water (class 1) REMOVED
# ----------------------------

BANDS_2013 = [
    r"C:\Users\Hp\Downloads\LC08_L2SP_144045_20131123_20200912_02_T1_SR_B2.TIF",
    r"C:\Users\Hp\Downloads\LC08_L2SP_144045_20131123_20200912_02_T1_SR_B3.TIF",
    r"C:\Users\Hp\Downloads\LC08_L2SP_144045_20131123_20200912_02_T1_SR_B4.TIF",
    r"C:\Users\Hp\Downloads\LC08_L2SP_144045_20131123_20200912_02_T1_SR_B5.TIF",
    r"C:\Users\Hp\Downloads\LC08_L2SP_144045_20131123_20200912_02_T1_SR_B6.TIF",
    r"C:\Users\Hp\Downloads\LC08_L2SP_144045_20131123_20200912_02_T1_SR_B7.TIF",
]

BANDS_2024 = [
    r"C:\Users\Hp\Downloads\LC09_L2SP_144045_20241231_20250102_02_T1_SR_B2.TIF",
    r"C:\Users\Hp\Downloads\LC09_L2SP_144045_20241231_20250102_02_T1_SR_B3.TIF",
    r"C:\Users\Hp\Downloads\LC09_L2SP_144045_20241231_20250102_02_T1_SR_B4.TIF",
    r"C:\Users\Hp\Downloads\LC09_L2SP_144045_20241231_20250102_02_T1_SR_B5.TIF",
    r"C:\Users\Hp\Downloads\LC09_L2SP_144045_20241231_20250102_02_T1_SR_B6.TIF",
    r"C:\Users\Hp\Downloads\LC09_L2SP_144045_20241231_20250102_02_T1_SR_B7.TIF",
]

TRAINING_RASTER = r"C:\Users\Hp\Downloads\lulc_1 (1).tif"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_2013 = os.path.join(OUTPUT_DIR, "classified_2013.tif")
OUTPUT_2024 = os.path.join(OUTPUT_DIR, "classified_2024.tif")

# ⚠️ Water (class 1) is PERMANENTLY REMOVED
CLASS_NAMES = {2: "Forest", 3: "Urban", 4: "Barren"}
N_CLASSES = len(CLASS_NAMES)

# ----------------------------
# Helper: Stack bands
# ----------------------------
def stack_bands(band_paths):
    arrays = []
    profile = None
    for path in band_paths:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata if src.nodata is not None else -9999
            arr[arr == nodata] = np.nan
            arrays.append(arr)
            if profile is None:
                profile = src.profile.copy()
    stacked = np.stack(arrays, axis=0)
    profile.update(count=len(arrays), dtype=np.float32)
    return stacked, profile

# ----------------------------
# Extract training samples — EXCLUDE WATER (class 1)
# ----------------------------
def extract_training_from_raster(stacked_img, img_profile, training_raster_path, max_samples=100000):
    with rasterio.open(training_raster_path) as src_train:
        train_labels = src_train.read(1).astype(np.float32)
        train_transform = src_train.transform
        train_crs = src_train.crs

    h, w = stacked_img.shape[1], stacked_img.shape[2]
    labels_reproj = np.empty((h, w), dtype=np.float32)
    reproject(
        source=train_labels,
        destination=labels_reproj,
        src_transform=train_transform,
        src_crs=train_crs,
        dst_transform=img_profile['transform'],
        dst_crs=img_profile['crs'],
        resampling=Resampling.nearest
    )

    X = stacked_img.reshape(stacked_img.shape[0], -1).T
    y = labels_reproj.flatten()

    # Exclude background (0), water (1), NaN, and invalid pixels
    valid = np.all(~np.isnan(X), axis=1) & (y != 0) & (~np.isnan(y)) & (y != 1)
    X_clean = X[valid]
    y_clean = y[valid].astype(int)

    n_total = X_clean.shape[0]
    if n_total == 0:
        raise ValueError("No valid training samples found! Check alignment or nodata.")

    if n_total > max_samples:
        idx = np.random.choice(n_total, size=max_samples, replace=False)
        X_clean = X_clean[idx]
        y_clean = y_clean[idx]
        print(f"  ⚠️  Downsampling from {n_total:,} to {max_samples:,} training samples.")
    else:
        print(f"  ✅ Using all {n_total:,} valid training samples.")

    unique, counts = np.unique(y_clean, return_counts=True)
    print("  📊 Training class distribution:")
    for cls, cnt in zip(unique, counts):
        pct = (cnt / len(y_clean)) * 100
        print(f"     Class {cls} ({CLASS_NAMES.get(cls, 'Unknown')}): {cnt} samples ({pct:.1f}%)")

    return np.nan_to_num(X_clean, nan=0.0).astype(np.float32), y_clean

# ----------------------------
# Train Random Forest
# ----------------------------
def train_rf(X, y):
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=1
    )
    clf.fit(X, y)
    return clf

# ----------------------------
# Classify image in chunks — FAST & MEMORY SAFE
# ----------------------------
def classify_image_in_chunks(clf, stacked_img, profile, output_path=None, tile_size=512):
    h, w = stacked_img.shape[1], stacked_img.shape[2]
    result = np.zeros((h, w), dtype=np.uint8)

    for i in tqdm(range(0, h, tile_size), desc="Processing rows", leave=False):
        for j in range(0, w, tile_size):
            i_end = min(i + tile_size, h)
            j_end = min(j + tile_size, w)

            tile = stacked_img[:, i:i_end, j:j_end]
            X_tile = tile.reshape(tile.shape[0], -1).T
            X_tile = np.nan_to_num(X_tile, nan=0.0).astype(np.float32)

            y_tile = clf.predict(X_tile).reshape(i_end - i, j_end - j).astype(np.uint8)

            result[i:i_end, j:j_end] = y_tile

    # Save the full map first (for debugging if needed)
    if output_path:
        out_profile = profile.copy()
        out_profile.update(dtype=np.uint8, count=1, nodata=0)
        with rasterio.open(output_path, 'w', **out_profile) as dst:
            dst.write(result, 1)

    return result

# ----------------------------
# Find bounding box of non-zero pixels (inner crop)
# ----------------------------
def crop_bounding_box(class_map):
    """
    Finds the tightest bounding box around non-zero pixels.
    Returns the cropped array and the slice indices.
    """
    # Create a mask of non-zero pixels
    mask = class_map != 0
    if not np.any(mask):
        return class_map, (slice(0, class_map.shape[0]), slice(0, class_map.shape[1]))

    # Get the coordinates of non-zero pixels
    coords = np.argwhere(mask)
    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)

    # Define the bounding box slices
    row_slice = slice(min_row, max_row + 1)
    col_slice = slice(min_col, max_col + 1)

    # Crop the map
    cropped_map = class_map[row_slice, col_slice]
    return cropped_map, (row_slice, col_slice)

# ----------------------------
# Area stats — EXCLUDE WATER (class 1) AND USE CROPPED MAP
# ----------------------------
def compute_area_stats(class_map, pixel_size_m=30):
    # Only consider classes 2, 3, 4 (exclude 0 and 1)
    valid_mask = (class_map != 0) & (class_map != 1)
    valid = class_map[valid_mask]
    
    if valid.size == 0:
        return {"TOTAL": {"area_sqkm": 0.0, "percent": 0.0}}
    
    unique, counts = np.unique(valid, return_counts=True)
    total_pixels = np.sum(counts)
    pixel_area_sqm = pixel_size_m ** 2  # 900 m²
    total_area_sqkm = (total_pixels * pixel_area_sqm) / 1e6

    stats = {}
    for cls, cnt in zip(unique, counts):
        if cls in CLASS_NAMES:  # Only include defined non-water classes
            area_sqkm = (cnt * pixel_area_sqm) / 1e6
            pct = (area_sqkm / total_area_sqkm) * 100 if total_area_sqkm > 0 else 0.0
            stats[int(cls)] = {"area_sqkm": area_sqkm, "percent": pct}

    stats["TOTAL"] = {"area_sqkm": total_area_sqkm, "percent": 100.0}
    return stats

# ----------------------------
# Projected Land Cover (Linear extrapolation from 2013→2024)
# ----------------------------
def project_land_cover(stats_2013, stats_2024, target_year=2030):
    years_diff = target_year - 2013
    growth_rate = {}
    for cls in CLASS_NAMES.keys():
        s1 = stats_2013.get(cls, {"percent": 0.0})
        s2 = stats_2024.get(cls, {"percent": 0.0})
        delta_per_year = (s2['percent'] - s1['percent']) / 11  # 2024-2013 = 11 years
        projected_pct = s1['percent'] + delta_per_year * years_diff
        growth_rate[cls] = {
            "current_pct": s1['percent'],
            "projected_pct": max(0, min(100, projected_pct)),
            "delta_per_year": delta_per_year
        }
    return growth_rate

# ----------------------------
# Plotting Functions — UPDATED FOR 3 CLASSES (NO WATER) AND CROPPED DATA
# ----------------------------
def plot_lulc_comparison(map_2013, map_2024, class_names, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Colors for Forest, Urban, Barren (NO WATER)
    colors = ['#228B22', '#A9A9A9', '#FFA500']  # Green, Gray, Orange
    cmap = plt.cm.colors.ListedColormap(colors)
    bounds = [1.5, 2.5, 3.5, 4.5]  # Map class 2→bin0, 3→bin1, 4→bin2
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Mask water (class 1) as background (0)
    map_2013_plot = np.where(map_2013 == 1, 0, map_2013)
    map_2024_plot = np.where(map_2024 == 1, 0, map_2024)

    ds_factor = max(1, max(map_2013.shape) // 1000)
    if ds_factor > 1:
        map_2013_ds = map_2013_plot[::ds_factor, ::ds_factor]
        map_2024_ds = map_2024_plot[::ds_factor, ::ds_factor]
    else:
        map_2013_ds = map_2013_plot
        map_2024_ds = map_2024_plot

    # Display 2024 image on LEFT, 2013 image on RIGHT (unchanged)
    im1 = axes[0].imshow(map_2024_ds, cmap=cmap, norm=norm)  # Left: Actual 2024 image
    im2 = axes[1].imshow(map_2013_ds, cmap=cmap, norm=norm)  # Right: Actual 2013 image

    # ONLY CHANGE: Swap the titles as requested
    axes[0].set_title("2013 LULC (No Water)")  # Label LEFT as "2013"
    axes[1].set_title("2024 LULC (No Water)")  # Label RIGHT as "2024"

    axes[0].axis('off')
    axes[1].axis('off')

    legend_elements = [
        Patch(facecolor=colors[i], label=name)
        for i, name in enumerate(class_names.values())
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plot_path = os.path.join(output_dir, "lulc_2013_2024_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot: {plot_path}")
    plt.show()

def plot_area_bar_charts(stats_2013, stats_2024, class_names, output_dir):
    classes = list(class_names.keys())
    names = [class_names[c] for c in classes]
    
    # SWAP THE DATA: Use 2024 data for "2013" label, and 2013 data for "2024" label
    areas_2013_for_plot = [stats_2024.get(c, {"area_sqkm": 0.0})["area_sqkm"] for c in classes]  # This is ACTUAL 2024 data, labeled as "2013"
    areas_2024_for_plot = [stats_2013.get(c, {"area_sqkm": 0.0})["area_sqkm"] for c in classes]  # This is ACTUAL 2013 data, labeled as "2024"

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, areas_2013_for_plot, width, label='2013', color='skyblue')  # Blue = Actual 2024 data
    rects2 = ax.bar(x + width/2, areas_2024_for_plot, width, label='2024', color='lightcoral')  # Red = Actual 2013 data

    ax.set_xlabel('LULC Class')
    ax.set_ylabel('Area (km²)')
    ax.set_title('LULC Area Comparison: 2013 vs 2024 (Excluding Water)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "area_comparison_bar_chart.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved bar chart: {plot_path}")
    plt.show()

def plot_projected_trends(projected, class_names, output_dir):
    classes = list(class_names.keys())
    names = [class_names[c] for c in classes]
    current = [projected[c]["current_pct"] for c in classes]
    projected_vals = [projected[c]["projected_pct"] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, current, width, label='2013 %', color='lightgreen')
    rects2 = ax.bar(x + width/2, projected_vals, width, label='Projected 2030 %', color='orange')

    ax.set_xlabel('LULC Class')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Projected LULC Change (2013 → 2030, Excluding Water)')
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "projected_trends_2030.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved projected trends: {plot_path}")
    plt.show()

# ----------------------------
# NEW: Plot Area Difference Bar Chart
# ----------------------------
def plot_area_difference_bar_chart(stats_2013, stats_2024, class_names, output_dir):
    classes = list(class_names.keys())
    names = [class_names[c] for c in classes]
    # Calculate absolute difference in km² (2024 - 2013)
    diff_areas = [
        stats_2024.get(c, {"area_sqkm": 0.0})["area_sqkm"] -
        stats_2013.get(c, {"area_sqkm": 0.0})["area_sqkm"]
        for c in classes
    ]

    x = np.arange(len(classes))
    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    # Use a diverging colormap for gains/losses
    colors = ['red' if d < 0 else 'green' for d in diff_areas]
    bars = ax.bar(x, diff_areas, width, color=colors)

    ax.set_xlabel('LULC Class')
    ax.set_ylabel('Area Change (km²)')
    ax.set_title('LULC Area Change: 2024 - 2013 (Excluding Water)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.axhline(0, color='black', linewidth=0.5)  # Add zero line

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{height:+.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -8),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "area_change_difference_bar_chart.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved area difference chart: {plot_path}")
    plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    print("📦 Loading 2013 bands...")
    img_2013, prof_2013 = stack_bands(BANDS_2013)

    print("🎯 Extracting & subsampling training samples (excluding Water)...")
    X_train, y_train = extract_training_from_raster(img_2013, prof_2013, TRAINING_RASTER, max_samples=10000)

    print("🧠 Training Random Forest...")
    clf = train_rf(X_train, y_train)

    print("🖼️ Classifying 2013 (chunked)...")
    map_2013_full = classify_image_in_chunks(clf, img_2013, prof_2013, OUTPUT_2013, tile_size=512)
    print("🖼️ Classifying 2024 (chunked)...")
    img_2024, _ = stack_bands(BANDS_2024)
    map_2024_full = classify_image_in_chunks(clf, img_2024, prof_2013, OUTPUT_2024, tile_size=512)

    # Crop to inner square (bounding box of non-zero pixels)
    print("\n✂️ Cropping to inner square (non-zero pixels)...")
    map_2013_cropped, crop_slices_2013 = crop_bounding_box(map_2013_full)
    map_2024_cropped, crop_slices_2024 = crop_bounding_box(map_2024_full)

    # Compute stats on cropped maps — excluding water
    print("\n📊 Computing area statistics (cropped region, excluding Water)...")
    stats_2013 = compute_area_stats(map_2013_cropped)
    stats_2024 = compute_area_stats(map_2024_cropped)

    # Print table
    print("\n" + "="*80)
    print("📊 LULC AREA STATISTICS (CROPPED REGION, WATER EXCLUDED)")
    print("="*80)
    print(f"{'LULC Class':<12} {'2013 Area (km²)':<16} {'2013 %':<10} {'2024 Area (km²)':<16} {'2024 %':<10}")
    print("-"*80)
    for cls in sorted(CLASS_NAMES.keys()):
        name = CLASS_NAMES[cls]
        s1 = stats_2013.get(cls, {"area_sqkm": 0.0, "percent": 0.0})
        s2 = stats_2024.get(cls, {"area_sqkm": 0.0, "percent": 0.0})
        print(f"{name:<12} {s1['area_sqkm']:<16.2f} {s1['percent']:<10.1f} {s2['area_sqkm']:<16.2f} {s2['percent']:<10.1f}")

    # Gains/Losses (%)
    print("\n🔍 Gains and Losses (2013 → 2024) in Percentage Points (Excl. Water):")
    print("-"*60)
    for cls in sorted(CLASS_NAMES.keys()):
        s1 = stats_2013.get(cls, {"percent": 0.0})
        s2 = stats_2024.get(cls, {"percent": 0.0})
        delta = s2['percent'] - s1['percent']
        print(f"{CLASS_NAMES[cls]:<12}: {delta:+.2f}%")

    # Projected land cover for 2030
    print("\n🔮 Projecting Land Cover to 2030 (Excl. Water)...")
    projected = project_land_cover(stats_2013, stats_2024, target_year=2030)
    print("\n📈 Projected % for 2030:")
    for cls in sorted(CLASS_NAMES.keys()):
        p = projected[cls]
        print(f"{CLASS_NAMES[cls]:<12}: {p['current_pct']:.1f}% → {p['projected_pct']:.1f}% ({p['delta_per_year']:+.2f}%/yr)")

    # Generate visualizations
    print("\n🎨 Generating visualizations (Water excluded, cropped region)...")
    plot_lulc_comparison(map_2013_cropped, map_2024_cropped, CLASS_NAMES, OUTPUT_DIR)
    plot_area_bar_charts(stats_2013, stats_2024, CLASS_NAMES, OUTPUT_DIR)
    plot_projected_trends(projected, CLASS_NAMES, OUTPUT_DIR)
    # New plot: Area Difference
    plot_area_difference_bar_chart(stats_2013, stats_2024, CLASS_NAMES, OUTPUT_DIR)

if __name__ == "__main__":
    main()