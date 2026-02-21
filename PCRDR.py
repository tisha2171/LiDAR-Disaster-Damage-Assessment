import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import laspy
import time
import gc
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import seaborn as sns
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')

PRE_LAZ_PATH = "data/pretry2.laz"
POST_LAZ_PATH = "data/posttry2.laz"

def load_laz_file(file_path):
    print(f"Loading {file_path}...")
    start_time = time.time()
    
    try:
        from laspy.compression import LazBackend
        las = laspy.read(file_path, laz_backend=LazBackend.LazrsBackend)
    except:
        try:
            las = laspy.read(file_path, laz_backend=LazBackend.LaszipBackend)
        except:
            las = laspy.read(file_path)
        
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    print(f"Loaded {len(points)} points in {time.time() - start_time:.2f} seconds")
    return points, las

def create_dsm(points, resolution=0.5, x_range=None, y_range=None):
    if x_range is None:
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    else:
        x_min, x_max = x_range
        
    if y_range is None:
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    else:
        y_min, y_max = y_range
    
    x_size = int((x_max - x_min) / resolution) + 1
    y_size = int((y_max - y_min) / resolution) + 1
    
    print(f"Creating DSM with dimensions: {y_size} x {x_size}")
    
    dsm = np.full((y_size, x_size), np.nan)
    
    batch_size = 250000
    num_batches = (len(points) + batch_size - 1) // batch_size
    
    for b in range(num_batches):
        if b % 10 == 0:
            print(f"Processing batch {b+1}/{num_batches}")
            
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, len(points))
        batch_points = points[start_idx:end_idx]
        
        x_indices = np.floor((batch_points[:, 0] - x_min) / resolution).astype(int)
        y_indices = np.floor((batch_points[:, 1] - y_min) / resolution).astype(int)
        
        valid_mask = (x_indices >= 0) & (x_indices < x_size) & (y_indices >= 0) & (y_indices < y_size)
        x_indices = x_indices[valid_mask]
        y_indices = y_indices[valid_mask]
        z_values = batch_points[valid_mask, 2]
        
        for i in range(len(x_indices)):
            if np.isnan(dsm[y_indices[i], x_indices[i]]) or z_values[i] > dsm[y_indices[i], x_indices[i]]:
                dsm[y_indices[i], x_indices[i]] = z_values[i]
        
        del batch_points, x_indices, y_indices, z_values, valid_mask
        gc.collect()
    
    return dsm, (x_min, y_min, resolution)

def simulate_building_footprints(pre_dsm, post_dsm, min_building_size=20, height_threshold=1.8, max_buildings=8000):
    print("Extracting building footprints...")
    start_time = time.time()
    
    ground_level = np.nanpercentile(pre_dsm, 10)
    print(f"Estimated ground level: {ground_level:.2f} m")
    
    building_mask = (pre_dsm - ground_level) > height_threshold
    
    chunk_size = 500
    height, width = building_mask.shape
    
    s = ndimage.generate_binary_structure(2, 2)
    
    labeled_buildings = np.zeros_like(building_mask, dtype=np.int32)
    current_label = 1
    
    print(f"Processing building mask in chunks...")
    
    for i in range(0, height, chunk_size):
        end_i = min(i + chunk_size, height)
        
        for j in range(0, width, chunk_size):
            end_j = min(j + chunk_size, width)
            
            chunk_mask = building_mask[i:end_i, j:end_j].copy()
            if not np.any(chunk_mask):
                continue
                
            chunk_labels, _ = ndimage.label(chunk_mask, structure=s)
            
            if np.max(chunk_labels) == 0:
                continue
                
            for label in range(1, np.max(chunk_labels) + 1):
                footprint = chunk_labels == label
                if np.sum(footprint) >= min_building_size:
                    labeled_buildings[i:end_i, j:end_j][footprint] = current_label
                    current_label += 1
            
            del chunk_mask, chunk_labels
            gc.collect()
    
    print(f"Labeling completed. Processing footprints...")
    
    max_buildings = min(max_buildings, np.max(labeled_buildings))
    footprints = []
    
    batch_size = 50
    num_batches = (max_buildings + batch_size - 1) // batch_size
    
    for b in range(num_batches):
        start_label = b * batch_size + 1
        end_label = min((b + 1) * batch_size, max_buildings) + 1
        
        if b % 5 == 0:
            print(f"Processing building batch {b+1}/{num_batches}")
        
        for label in range(start_label, end_label):
            footprint = labeled_buildings == label
            if np.sum(footprint) >= min_building_size:
                footprints.append(csr_matrix(footprint))
        
        gc.collect()
    
    print(f"Identified {len(footprints)} potential buildings in {time.time() - start_time:.2f} seconds")
    return footprints

def calculate_building_features(pre_dsm, post_dsm, footprints):
    print("Calculating building features...")
    start_time = time.time()
    
    features = []
    batch_size = 20
    num_batches = (len(footprints) + batch_size - 1) // batch_size
    
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, len(footprints))
        
        if b % 10 == 0:
            print(f"Processing building batch {b+1}/{num_batches}")
        
        for i in range(start_idx, end_idx):
            footprint = footprints[i].toarray().astype(bool)
            
            pre_heights = pre_dsm[footprint]
            post_heights = post_dsm[footprint]
            
            if np.sum(~np.isnan(pre_heights)) == 0 or np.sum(~np.isnan(post_heights)) == 0:
                continue
            
            delta_h = np.nanmean(post_heights - pre_heights)
            std_dev = np.nanstd(post_heights - pre_heights)
            
            valid_mask = ~np.isnan(pre_heights) & ~np.isnan(post_heights)
            if np.sum(valid_mask) < 2:
                continue
                
            pre_valid = pre_heights[valid_mask]
            post_valid = post_heights[valid_mask]
            
            try:
                corr, _ = pearsonr(pre_valid, post_valid)
            except:
                corr = 0
                
            rows, cols = np.where(footprint)
            centroid_y = np.mean(rows)
            centroid_x = np.mean(cols)
            
            max_height_diff = np.nanmax(np.abs(post_heights - pre_heights))
            min_height_diff = np.nanmin(post_heights - pre_heights)
            height_range = np.nanmax(pre_heights) - np.nanmin(pre_heights)
            
            features.append({
                'delta_h': delta_h,
                'std_dev': std_dev,
                'correlation': corr,
                'max_height_diff': max_height_diff,
                'min_height_diff': min_height_diff,
                'height_range': height_range,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'footprint': footprint,
                'area': np.sum(footprint)
            })
            
            del footprint, pre_heights, post_heights, valid_mask
            if 'pre_valid' in locals(): del pre_valid
            if 'post_valid' in locals(): del post_valid
        
        gc.collect()
    
    print(f"Calculated features for {len(features)} buildings in {time.time() - start_time:.2f} seconds")
    return features

def detect_collapsed_buildings(features, threshold=-0.5):
    collapsed = []
    non_collapsed = []
    
    for building in features:
        if building['delta_h'] < threshold:
            building['status'] = 'collapsed'
            collapsed.append(building)
        else:
            building['status'] = 'non_collapsed'
            non_collapsed.append(building)
    
    print(f"Detected {len(collapsed)} collapsed buildings and {len(non_collapsed)} non-collapsed buildings")
    return collapsed, non_collapsed

def prepare_features_for_ml(features):
    X = np.array([
        [f['delta_h'], f['std_dev'], f['correlation'], 
         f['max_height_diff'], f['min_height_diff'], f['height_range'], 
         f['area']] for f in features
    ])
    y = np.array([1 if f['status'] == 'collapsed' else 0 for f in features])
    
    return X, y

def train_ensemble_classifier(features):
    X, y = prepare_features_for_ml(features)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
    logistic_classifier = LogisticRegression(max_iter=1000, random_state=42)
    xgb_classifier = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_classifier),
            ('svm', svm_classifier),
            ('logistic', logistic_classifier),
            ('xgb', xgb_classifier)
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train_scaled, y_train)
    
    y_pred = ensemble.predict(X_test_scaled)
    y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    plt.figure(figsize=(15, 5))
    feature_names = [
        'Delta H', 'Std Dev', 'Correlation', 
        'Max Height Diff', 'Min Height Diff', 
        'Height Range', 'Area'
    ]
    importances = ensemble.named_estimators_['rf'].feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)

    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
               
    
    return ensemble, scaler, X_test, y_test, y_pred, y_pred_proba 

def load_and_process_data(pre_file, post_file, resolution=1.0):
    pre_points, pre_las = load_laz_file(pre_file)
    post_points, post_las = load_laz_file(post_file)
    
    x_min = max(np.min(pre_points[:, 0]), np.min(post_points[:, 0]))
    x_max = min(np.max(pre_points[:, 0]), np.max(post_points[:, 0]))
    y_min = max(np.min(pre_points[:, 1]), np.min(post_points[:, 1]))
    y_max = min(np.max(pre_points[:, 1]), np.max(post_points[:, 1]))
    
    print(f"Common bounds: X({x_min:.2f}, {x_max:.2f}), Y({y_min:.2f}, {y_max:.2f})")
    
    pre_dsm, dsm_params = create_dsm(pre_points, resolution, (x_min, x_max), (y_min, y_max))
    post_dsm, _ = create_dsm(post_points, resolution, (x_min, x_max), (y_min, y_max))
    
    return pre_dsm, post_dsm, dsm_params, pre_points, post_points

def visualize_dsm_difference(pre_dsm, post_dsm):
    # Create figure with better proportions
    plt.figure(figsize=(15, 10), dpi=300)
    
    # Create subplot with specific size ratio
    plt.subplot(111)
    
    # Calculate difference
    diff = post_dsm - pre_dsm
    masked_diff = np.ma.masked_invalid(diff)
    
    # Add shading for 3D effect
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=45)
    shaded_diff = ls.shade(masked_diff, plt.cm.RdBu_r, vmin=-5, vmax=5)
    
    # Plot with enhanced styling but smaller markers/fonts
    im = plt.imshow(masked_diff, cmap='RdBu_r', vmin=-5, vmax=5)
    
    # Add contour lines with smaller linewidths
    contours = plt.contour(masked_diff, colors='black', alpha=0.2, linewidths=0.3)
    plt.clabel(contours, inline=True, fontsize=6, fmt='%1.1f')
    
    # Enhanced colorbar with adjusted size
    cbar = plt.colorbar(im, label='Height Difference (m)', pad=0.01, fraction=0.046)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Height Difference (m)', size=10)
    
    # Improved title with adjusted size
    plt.title('Digital Surface Model Height Difference', 
              fontsize=12, pad=10)
    
    # Add grid with subtle appearance
    plt.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
    
    # Add north arrow with adjusted size
    plt.annotate('N', xy=(0.02, 0.98), xycoords='axes fraction', 
                fontsize=8, ha='center', va='center')
    plt.arrow(0.02, 0.95, 0, 0.02, head_width=0.008, 
              head_length=0.008, fc='k', ec='k')
    
    # Adjust layout with specific margins
    plt.tight_layout(pad=1.5)
    
    # Save with high DPI
    plt.savefig('dsm_difference.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def visualize_building_status(pre_dsm, post_dsm, collapsed_buildings, non_collapsed_buildings):
    # Create figure with higher DPI
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Calculate DSM difference for background context
    dsm_diff = post_dsm - pre_dsm
    masked_diff = np.ma.masked_invalid(dsm_diff)
    
    # Plot DSM difference as background
    plt.imshow(masked_diff, cmap='Greys', alpha=0.3, vmin=-5, vmax=5)
    
    # Plot buildings with improved visualization
    for building in non_collapsed_buildings:
        y, x = np.where(building['footprint'])
        # Plot intact buildings as filled green polygons
        plt.fill(x, y, facecolor='green', edgecolor='darkgreen', 
                alpha=0.6, linewidth=1, label='_nolegend_')
        # Add a circle marker at the centroid
        plt.plot(building['centroid_x'], building['centroid_y'], 
                'o', color='darkgreen', markersize=8, markeredgecolor='white')
    
    for building in collapsed_buildings:
        y, x = np.where(building['footprint'])
        # Plot collapsed buildings as red polygons with hatching
        plt.fill(x, y, facecolor='red', edgecolor='darkred', 
                alpha=0.6, linewidth=1, hatch='xx', label='_nolegend_')
        # Add an X marker at the centroid
        plt.plot(building['centroid_x'], building['centroid_y'], 
                'X', color='darkred', markersize=10, markeredgecolor='white')
    
    # Enhance the plot
    plt.title('Building Damage Assessment', fontsize=14, pad=20, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.2, linestyle='--')
    
    # Improved legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor='green', alpha=0.6, 
                     edgecolor='darkgreen', label='Intact Building'),
        plt.Rectangle((0,0), 1, 1, facecolor='red', alpha=0.6, 
                     edgecolor='darkred', hatch='xx', label='Collapsed Building'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', 
                  markersize=10, label='Intact Building Centroid'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='darkred', 
                  markersize=10, label='Collapsed Building Centroid')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', 
              title='Building Status', title_fontsize=12)
    
    # Add colorbar for height difference
    cbar = plt.colorbar(label='Height Difference (m)', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    # Add scale bar (if spatial resolution is known)
    # from matplotlib_scalebar.scalebar import ScaleBar
    # plt.gca().add_artist(ScaleBar(dx=1))  # dx should be your spatial resolution
    
    # Add north arrow
    plt.annotate('N', xy=(0.02, 0.98), xycoords='axes fraction', 
                fontsize=12, ha='center', va='center')
    plt.arrow(0.02, 0.95, 0, 0.02, head_width=0.01, 
              head_length=0.01, fc='k', ec='k')
    
    # Improve layout
    plt.tight_layout()
    
    # Save with high resolution
    plt.savefig('building_damage_assessment.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def visualize_height_profiles(pre_dsm, post_dsm, collapsed_buildings, non_collapsed_buildings):
    plt.figure(figsize=(12, 6))
    
    # Plot height profiles for a sample of collapsed buildings
    for i, building in enumerate(collapsed_buildings[:5]):
        y, x = np.where(building['footprint'])
        pre_heights = pre_dsm[building['footprint']]
        post_heights = post_dsm[building['footprint']]
        
        valid_mask = ~np.isnan(pre_heights) & ~np.isnan(post_heights)
        if np.sum(valid_mask) < 10:
            continue
            
        pre_valid = pre_heights[valid_mask]
        post_valid = post_heights[valid_mask]
        
        plt.subplot(2, 5, i+1)
        plt.scatter(range(len(pre_valid)), pre_valid, color='blue', alpha=0.7, s=5, label='Pre-earthquake')
        plt.scatter(range(len(post_valid)), post_valid, color='red', alpha=0.7, s=5, label='Post-earthquake')
        plt.title(f'Collapsed Building {i+1}')
        if i == 0:
            plt.legend()
        plt.ylabel('Height (m)')
    
    # Plot height profiles for a sample of non-collapsed buildings
    for i, building in enumerate(non_collapsed_buildings[:5]):
        y, x = np.where(building['footprint'])
        pre_heights = pre_dsm[building['footprint']]
        post_heights = post_dsm[building['footprint']]
        
        valid_mask = ~np.isnan(pre_heights) & ~np.isnan(post_heights)
        if np.sum(valid_mask) < 10:
            continue
            
        pre_valid = pre_heights[valid_mask]
        post_valid = post_heights[valid_mask]
        
        plt.subplot(2, 5, i+6)
        plt.scatter(range(len(pre_valid)), pre_valid, color='blue', alpha=0.7, s=5, label='Pre-earthquake')
        plt.scatter(range(len(post_valid)), post_valid, color='red', alpha=0.7, s=5, label='Post-earthquake')
        plt.title(f'Non-collapsed Building {i+1}')
        if i == 0:
            plt.legend()
        plt.ylabel('Height (m)')
        plt.xlabel('Point Index')
    
    plt.tight_layout()
    plt.savefig('height_profiles.png')
    plt.show()
    plt.close()

def visualize_feature_distributions(features, collapsed_buildings, non_collapsed_buildings):
    # Extract features for collapsed and non-collapsed buildings
    collapsed_features = np.array([
        [f['delta_h'], f['std_dev'], f['correlation'], 
         f['max_height_diff'], f['min_height_diff'], f['height_range'], 
         f['area']] for f in collapsed_buildings
    ])
    
    non_collapsed_features = np.array([
        [f['delta_h'], f['std_dev'], f['correlation'], 
         f['max_height_diff'], f['min_height_diff'], f['height_range'], 
         f['area']] for f in non_collapsed_buildings
    ])
    
    feature_names = [
        'Delta H', 'Std Dev', 'Correlation', 
        'Max Height Diff', 'Min Height Diff', 
        'Height Range', 'Area'
    ]
    
    # Create distribution plots for each feature
    plt.figure(figsize=(15, 10))
    
    for i, name in enumerate(feature_names):
        plt.subplot(2, 4, i+1)
        
        if len(collapsed_features) > 0:
            sns.kdeplot(collapsed_features[:, i], label='Collapsed', color='red')
        
        if len(non_collapsed_features) > 0:
            sns.kdeplot(non_collapsed_features[:, i], label='Non-collapsed', color='green')
        
        plt.title(name)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.show()
    plt.close()

def main():
    pre_dsm, post_dsm, dsm_params, pre_points, post_points = load_and_process_data(PRE_LAZ_PATH, POST_LAZ_PATH, resolution=1.0)
    
    footprints = simulate_building_footprints(pre_dsm, post_dsm)
    features = calculate_building_features(pre_dsm, post_dsm, footprints)
    
    collapsed_buildings, non_collapsed_buildings = detect_collapsed_buildings(features)
    
    ensemble, scaler, X_test, y_test, y_pred, y_pred_proba = train_ensemble_classifier(features)
    
    visualize_dsm_difference(pre_dsm, post_dsm)
    visualize_building_status(pre_dsm, post_dsm, collapsed_buildings, non_collapsed_buildings)
    visualize_height_profiles(pre_dsm, post_dsm, collapsed_buildings, non_collapsed_buildings)
    visualize_feature_distributions(features, collapsed_buildings, non_collapsed_buildings)
    print("Analysis complete")

if __name__ == "__main__":
    main()