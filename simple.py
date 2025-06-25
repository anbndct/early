import streamlit as st
from streamlit_option_menu import option_menu
import nibabel as nib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import tempfile
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes

# ===============================================================
# 🔧 CUSTOM LAYERS & CONFIGURATION
# ===============================================================

# Custom layer for CNN
class SpatialPyramidPooling3D(tf.keras.layers.Layer):
    def __init__(self, pool_list=[1, 2, 4], **kwargs):
        super(SpatialPyramidPooling3D, self).__init__(**kwargs)
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i**3 for i in pool_list])

    def build(self, input_shape):
        super(SpatialPyramidPooling3D, self).build(input_shape)
        self.nb_channels = input_shape[-1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = super(SpatialPyramidPooling3D, self).get_config()
        config.update({'pool_list': self.pool_list})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x, **kwargs):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        height = tf.cast(input_shape[1], tf.float32)
        width = tf.cast(input_shape[2], tf.float32)
        depth = tf.cast(input_shape[3], tf.float32)
        channels = input_shape[4]

        outputs = []
        for pool_size in self.pool_list:
            pool_height = tf.cast(height / pool_size, tf.int32)
            pool_width = tf.cast(width / pool_size, tf.int32)
            pool_depth = tf.cast(depth / pool_size, tf.int32)

            pool_height = tf.maximum(pool_height, 1)
            pool_width = tf.maximum(pool_width, 1)
            pool_depth = tf.maximum(pool_depth, 1)

            for i in range(pool_size):
                for j in range(pool_size):
                    for k in range(pool_size):
                        h_start = tf.cast(i * height / pool_size, tf.int32)
                        h_end = tf.cast((i + 1) * height / pool_size, tf.int32)
                        w_start = tf.cast(j * width / pool_size, tf.int32)
                        w_end = tf.cast((j + 1) * width / pool_size, tf.int32)
                        d_start = tf.cast(k * depth / pool_size, tf.int32)
                        d_end = tf.cast((k + 1) * depth / pool_size, tf.int32)

                        h_end = tf.maximum(h_end, h_start + 1)
                        w_end = tf.maximum(w_end, w_start + 1)
                        d_end = tf.maximum(d_end, d_start + 1)

                        region = x[:, h_start:h_end, w_start:w_end, d_start:d_end, :]
                        pooled = tf.reduce_max(region, axis=[1, 2, 3])
                        outputs.append(pooled)

        output = tf.concat(outputs, axis=-1)
        return output

CUSTOM_OBJECTS = {'SpatialPyramidPooling3D': SpatialPyramidPooling3D}

# ===============================================================
# 📊 CONFIGURATION
# ===============================================================

# FCN Configuration
FCN_PATCH_SHAPE = (16, 16, 10)
FCN_STRIDE = (4, 4, 2)
FCN_BATCH_SIZE = 512
FCN_THRESHOLD = 0.85
FCN_MIN_DISTANCE = 12
FCN_MIN_CLUSTER_SIZE = 3

# CNN Configuration
CNN_PATCH_SHAPE = (20, 20, 16)
CNN_PATCH_X, CNN_PATCH_Y, CNN_PATCH_Z = 10, 10, 8
CNN_THRESHOLD = 0.5
GT_TOLERANCE = 7

# Dataset Configuration
HUGGINGFACE_BASE_URL = "https://huggingface.co/datasets/anbndct/NII_MRI_CMB/resolve/main"
FCN_RESULTS_URL = "https://huggingface.co/datasets/anbndct/NII_MRI_CMB/resolve/main/fcn_results.npy"
FCN_MODEL_URL = "https://huggingface.co/datasets/anbndct/NII_MRI_CMB/resolve/main/fcn_precision_focused_best.h5"
CNN_MODEL_URL = "https://huggingface.co/datasets/anbndct/NII_MRI_CMB/resolve/main/stage2_cnn_final.h5"

# Dataset metadata - matches your actual file naming (01.nii, 02.nii, etc.)
DATASET_INFO = {
    1: {"nii": "01.nii", "gt": "01.mat", "size_mb": 315},
    2: {"nii": "02.nii", "gt": "02.mat", "size_mb": 315},
    3: {"nii": "03.nii", "gt": "03.mat", "size_mb": 315},
    4: {"nii": "04.nii", "gt": "04.mat", "size_mb": 315},
    5: {"nii": "05.nii", "gt": "05.mat", "size_mb": 315},
    6: {"nii": "06.nii", "gt": "06.mat", "size_mb": 315},
    7: {"nii": "07.nii", "gt": "07.mat", "size_mb": 315},
    8: {"nii": "08.nii", "gt": "08.mat", "size_mb": 315},
    9: {"nii": "09.nii", "gt": "09.mat", "size_mb": 315},
    10: {"nii": "10.nii", "gt": "10.mat", "size_mb": 315},
    11: {"nii": "11.nii", "gt": "11.mat", "size_mb": 315},
    12: {"nii": "12.nii", "gt": "12.mat", "size_mb": 315},
    13: {"nii": "13.nii", "gt": "13.mat", "size_mb": 315},
    14: {"nii": "14.nii", "gt": "14.mat", "size_mb": 315},
    15: {"nii": "15.nii", "gt": "15.mat", "size_mb": 315},
    16: {"nii": "16.nii", "gt": "16.mat", "size_mb": 315},
    17: {"nii": "17.nii", "gt": "17.mat", "size_mb": 315},
    18: {"nii": "18.nii", "gt": "18.mat", "size_mb": 315},
    19: {"nii": "19.nii", "gt": "19.mat", "size_mb": 315},
    20: {"nii": "20.nii", "gt": "20.mat", "size_mb": 315}
}

# ===============================================================
# 🎨 CSS STYLING
# ===============================================================

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #D36BA3, #974578);
    color: white;
}
.circle-img {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    object-fit: cover;
    object-position: center;
}
.detection-card {
    background-color: white;
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.metric-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin: 5px;
}
.team-card {
    background-color: white;
    border-radius: 18px;
    padding: 1rem;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
}
.section-title {
    color: white;
    font-size: 22px;
    font-weight: bold;
    margin-top: 2rem;
}
.section-headline {
    font-size: 30px;
    font-weight: 800;
    color: white;
    margin-top: 1rem;
    margin-bottom: 0.2rem;
}
.section-subheadline {
    font-size: 20px;
    font-weight: 400;
    color: white;
    margin-bottom: 1.5rem;
}
.text-white {
    color: white;
    font-size: 18px;
}
ul {
    padding-left: 1.2rem;
}
.stChatFloatingInputContainer {
    background-color: #f0f2f6;
}
.stTextInput input {
    color: #333333 !important;
}
.stChatMessage {
    border-radius: 15px !important;
    padding: 12px !important;
    margin: 8px 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ===============================================================
# 📱 SIDEBAR MENU
# ===============================================================

with st.sidebar:
    selected = option_menu(
        "Cerebral Microbleeds Detection",
        ["Home", "Get to Know Microbleeds", "Ground Truth", "CMB Detection", "Chatbot", "Project FAQ"],
        default_index=0
    )

# ===============================================================
# 🔧 UTILITY FUNCTIONS
# ===============================================================

@st.cache_data(ttl=7200)  # Cache for 2 hours
def load_cmb_volume_data(volume_id):
    """Load specific CMB volume and ground truth data"""
    try:
        vol_str = f"{volume_id:02d}"
        nii_url = f"{HUGGINGFACE_BASE_URL}/{vol_str}.nii"
        mat_url = f"{HUGGINGFACE_BASE_URL}/{vol_str}.mat"
        
        with st.status(f"Loading Volume {volume_id}...", expanded=True) as status:
            # Download NII file
            st.write(f"📥 Downloading {vol_str}.nii (~315MB)...")
            
            nii_response = requests.get(nii_url, stream=True)
            nii_response.raise_for_status()
            
            # Progress bar for NII download
            total_size = int(nii_response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            
            nii_data = b""
            downloaded = 0
            
            for chunk in nii_response.iter_content(chunk_size=8192):
                if chunk:
                    nii_data += chunk
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(downloaded / total_size)
            
            progress_bar.empty()
            
            # Download MAT file
            st.write(f"📥 Downloading {vol_str}.mat...")
            mat_response = requests.get(mat_url)
            mat_response.raise_for_status()
            
            # Load data from bytes
            st.write("🔄 Loading volume data...")
            
            # Save to temp files and load
            with tempfile.NamedTemporaryFile(suffix='.nii') as nii_temp:
                nii_temp.write(nii_data)
                nii_temp.flush()
                img = nib.load(nii_temp.name)
                volume_data = img.get_fdata()
            
            with tempfile.NamedTemporaryFile(suffix='.mat') as mat_temp:
                mat_temp.write(mat_response.content)
                mat_temp.flush()
                gt_data = sio.loadmat(mat_temp.name)
            
            status.update(label=f"✅ Volume {volume_id} loaded successfully!", state="complete")
            
            return volume_data, gt_data
            
    except Exception as e:
        st.error(f"Error loading Volume {volume_id}: {str(e)}")
        return None, None

@st.cache_resource
def load_fcn_model():
    """Load FCN model from Hugging Face"""
    try:
        st.info("📥 Loading FCN model from Hugging Face...")
        response = requests.get(FCN_MODEL_URL)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix='.h5') as temp_file:
            temp_file.write(response.content)
            temp_file.flush()
            model = load_model(temp_file.name)
        
        st.success("✅ FCN model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Failed to load FCN model: {str(e)}")
        return None

@st.cache_resource
def load_cnn_model():
    """Load CNN model from Hugging Face"""
    try:
        st.info("📥 Loading CNN model from Hugging Face...")
        response = requests.get(CNN_MODEL_URL)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix='.h5') as temp_file:
            temp_file.write(response.content)
            temp_file.flush()
            model = load_model(temp_file.name, custom_objects=CUSTOM_OBJECTS)
        
        st.success("✅ CNN model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Failed to load CNN model: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_fcn_results():
    """Load pre-computed FCN results from Hugging Face"""
    try:
        st.info("📥 Loading pre-computed FCN results...")
        response = requests.get(FCN_RESULTS_URL)
        response.raise_for_status()
        
        fcn_results = np.load(io.BytesIO(response.content), allow_pickle=True).item()
        st.success(f"✅ FCN results loaded: {len(fcn_results)} volumes")
        return fcn_results
        
    except Exception as e:
        st.error(f"❌ Failed to load FCN results: {str(e)}")
        return None
    """Analyze uploaded FCN .npy file to extract metadata"""
    try:
        fcn_results = np.load(uploaded_file, allow_pickle=True).item()
        
        if not isinstance(fcn_results, dict):
            st.error("❌ Invalid .npy format - expected dictionary structure")
            return None, None, None
        
        volume_info = []
        total_candidates = 0
        
        for vol_id, vol_data in fcn_results.items():
            try:
                vol_info = vol_data.get('vol_info', {})
                candidates = vol_data.get('fcn_candidates', [])
                
                vol_name = vol_info.get('name', f'volume_{vol_id}')
                vol_shape = vol_data.get('volume_shape', vol_info.get('shape', 'Unknown'))
                candidates_count = len(candidates)
                
                candidate_scores = [c.get('score', 0) for c in candidates if isinstance(c, dict)]
                avg_score = np.mean(candidate_scores) if candidate_scores else 0
                
                volume_info.append({
                    'Volume ID': vol_id,
                    'Volume Name': vol_name,
                    'Shape': str(vol_shape),
                    'FCN Candidates': candidates_count,
                    'Avg Score': f"{avg_score:.3f}"
                })
                
                total_candidates += candidates_count
                
            except Exception as e:
                st.warning(f"⚠️ Issue processing volume {vol_id}: {str(e)}")
                continue
        
        processing_summary = {
            'total_volumes': len(volume_info),
            'total_candidates': total_candidates,
            'avg_candidates_per_volume': total_candidates / len(volume_info) if volume_info else 0,
            'volume_ids': sorted([info['Volume ID'] for info in volume_info]),
            'volume_range': f"{min([info['Volume ID'] for info in volume_info])}-{max([info['Volume ID'] for info in volume_info])}" if volume_info else "N/A"
        }
        
        return fcn_results, volume_info, processing_summary
        
    except Exception as e:
        st.error(f"❌ Error analyzing .npy file: {str(e)}")
        return None, None, None

# ===============================================================
# 🔧 FCN FUNCTIONS (for real-time inference)
# ===============================================================

def create_brain_mask(volume):
    """Create brain mask"""
    threshold_mask = volume > 0.1
    filled_mask = binary_fill_holes(threshold_mask)
    eroded_mask = binary_erosion(filled_mask, iterations=2)
    brain_mask = binary_dilation(eroded_mask, iterations=4)
    return brain_mask

def fcn_inference(model, volume):
    """FCN sliding window inference with progress tracking"""
    H, W, D = volume.shape
    pH, pW, pD = FCN_PATCH_SHAPE
    sH, sW, sD = FCN_STRIDE

    score_map = np.zeros((H, W, D), dtype=np.float32)
    count_map = np.zeros((H, W, D), dtype=np.float32)

    # Calculate total patches
    total_patches = 0
    patch_coords = []
    for x in range(0, H-pH+1, sH):
        for y in range(0, W-pW+1, sW):
            for z in range(0, D-pD+1, sD):
                patch_coords.append((x, y, z))
                total_patches += 1

    st.info(f"🔍 FCN processing {total_patches:,} patches with batch size {FCN_BATCH_SIZE}")
    
    # Estimate processing time
    estimated_time = (total_patches / FCN_BATCH_SIZE) * 0.1  # ~0.1 seconds per batch
    st.info(f"⏱️ Estimated processing time: ~{estimated_time/60:.1f} minutes")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_patches = 0
    
    for batch_start in range(0, len(patch_coords), FCN_BATCH_SIZE):
        batch_end = min(batch_start + FCN_BATCH_SIZE, len(patch_coords))
        batch_coords = patch_coords[batch_start:batch_end]

        batch_patches = np.zeros((len(batch_coords), pH, pW, pD, 1), dtype=np.float32)
        valid_indices = []

        for i, (x, y, z) in enumerate(batch_coords):
            patch = volume[x:x+pH, y:y+pW, z:z+pD]
            if patch.shape == FCN_PATCH_SHAPE:
                batch_patches[i, :, :, :, 0] = patch
                valid_indices.append(i)

        if valid_indices:
            valid_patches = batch_patches[valid_indices]
            predictions = model.predict(valid_patches, verbose=0)

            for idx_in_valid, i in enumerate(valid_indices):
                x, y, z = batch_coords[i]
                cmb_prob = predictions[idx_in_valid, 1]

                cx, cy, cz = x + pH//2, y + pW//2, z + pD//2
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            ax, ay, az = cx + dx, cy + dy, cz + dz
                            if 0 <= ax < H and 0 <= ay < W and 0 <= az < D:
                                weight = 1.0 if (dx == 0 and dy == 0 and dz == 0) else 0.3
                                score_map[ax, ay, az] += cmb_prob * weight
                                count_map[ax, ay, az] += weight

        processed_patches += len(batch_coords)
        progress = processed_patches / total_patches
        progress_bar.progress(progress)
        status_text.text(f"FCN Progress: {processed_patches:,}/{total_patches:,} patches ({progress*100:.1f}%)")

    progress_bar.empty()
    status_text.empty()
    
    score_map = np.divide(score_map, count_map, out=np.zeros_like(score_map), where=count_map>0)
    return score_map

def fcn_clustering_nms(score_map):
    """FCN clustering and NMS"""
    binary_map = score_map > FCN_THRESHOLD
    if binary_map.sum() == 0:
        return []

    coords = np.where(binary_map)
    scores = score_map[coords]
    coord_array = np.column_stack([coords[0], coords[1], coords[2]])

    clustering = DBSCAN(eps=3, min_samples=FCN_MIN_CLUSTER_SIZE).fit(coord_array)

    candidates = []
    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:
            continue

        cluster_mask = clustering.labels_ == cluster_id
        cluster_coords = coord_array[cluster_mask]
        cluster_scores = scores[cluster_mask]

        if len(cluster_coords) >= FCN_MIN_CLUSTER_SIZE:
            weights = cluster_scores / cluster_scores.sum()
            center_coord = np.average(cluster_coords, axis=0, weights=weights)
            max_score = cluster_scores.max()

            # NMS
            too_close = False
            for existing in candidates:
                ex, ey, ez = existing['coordinate']
                dist = np.sqrt(np.sum((center_coord - np.array([ex, ey, ez]))**2))
                if dist < FCN_MIN_DISTANCE:
                    if max_score > existing['score']:
                        candidates.remove(existing)
                    else:
                        too_close = True
                    break

            if not too_close:
                candidates.append({
                    'coordinate': [int(round(center_coord[0])),
                                 int(round(center_coord[1])),
                                 int(round(center_coord[2]))],
                    'score': float(max_score),
                    'cluster_size': len(cluster_coords)
                })

    return sorted(candidates, key=lambda x: x['score'], reverse=True)

# ===============================================================
# 🔧 CNN FUNCTIONS
# ===============================================================

def extract_cnn_patches(volume, fcn_candidates):
    """Extract CNN patches from FCN candidates"""
    valid_patches = []
    valid_candidates = []

    for candidate in fcn_candidates:
        coord = candidate['coordinate']
        cx, cy, cz = int(coord[0]), int(coord[1]), int(coord[2])

        x_start, x_end = cx - CNN_PATCH_X, cx + CNN_PATCH_X
        y_start, y_end = cy - CNN_PATCH_Y, cy + CNN_PATCH_Y
        z_start, z_end = cz - CNN_PATCH_Z, cz + CNN_PATCH_Z

        if (x_start >= 0 and x_end <= volume.shape[0] and
            y_start >= 0 and y_end <= volume.shape[1] and
            z_start >= 0 and z_end <= volume.shape[2]):

            patch = volume[x_start:x_end, y_start:y_end, z_start:z_end]
            if patch.shape == CNN_PATCH_SHAPE:
                valid_patches.append(patch)
                valid_candidates.append(candidate)

    if len(valid_patches) == 0:
        return None, []

    X = np.array(valid_patches)[..., np.newaxis]
    return X, valid_candidates

def cnn_inference(model, X_patches, candidates):
    """CNN inference with filtering"""
    if X_patches is None or len(X_patches) == 0:
        return [], [], []

    predictions = model.predict(X_patches, batch_size=32, verbose=0)
    cmb_probs = predictions[:, 1]

    final_candidates = []
    rejected_candidates = []

    for i, candidate in enumerate(candidates):
        cmb_prob = cmb_probs[i]

        enhanced_candidate = candidate.copy()
        enhanced_candidate['fcn_score'] = candidate.get('score', 1.0)
        enhanced_candidate['cnn_score'] = float(cmb_prob)
        enhanced_candidate['final_score'] = float(cmb_prob)

        if cmb_prob >= CNN_THRESHOLD:
            final_candidates.append(enhanced_candidate)
        else:
            rejected_candidates.append(enhanced_candidate)

    return final_candidates, rejected_candidates, cmb_probs

def create_detection_visualization(volume, final_candidates, rejected_candidates, slice_idx, gt_coords=None):
    """Create detection visualization with optional ground truth"""
    fig, axes = plt.subplots(1, 4 if gt_coords is not None else 3, figsize=(24 if gt_coords is not None else 18, 6))
    
    # Original slice
    axes[0].imshow(volume[:, :, slice_idx], cmap='gray')
    axes[0].set_title(f'Original - Slice {slice_idx}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Final detections
    axes[1].imshow(volume[:, :, slice_idx], cmap='gray')
    final_in_slice = [c for c in final_candidates if int(c['coordinate'][2]) == slice_idx]
    for i, candidate in enumerate(final_in_slice):
        cx, cy, cz = candidate['coordinate']
        circle = plt.Circle((cy, cx), 5, color='red', fill=False, linewidth=3)
        axes[1].add_patch(circle)
        axes[1].text(cy+6, cx+6, f"{candidate['cnn_score']:.2f}", 
                    color='red', fontweight='bold', fontsize=10)
    axes[1].set_title(f'Final Detections ({len(final_in_slice)} in slice)', 
                     fontsize=14, fontweight='bold', color='red')
    axes[1].axis('off')
    
    # All candidates (final + rejected)
    axes[2].imshow(volume[:, :, slice_idx], cmap='gray')
    for candidate in final_in_slice:
        cx, cy, cz = candidate['coordinate']
        circle = plt.Circle((cy, cx), 5, color='red', fill=False, linewidth=3)
        axes[2].add_patch(circle)
    
    rejected_in_slice = [c for c in rejected_candidates if int(c['coordinate'][2]) == slice_idx]
    for candidate in rejected_in_slice:
        cx, cy, cz = candidate['coordinate']
        circle = plt.Circle((cy, cx), 4, color='blue', fill=False, linewidth=1, alpha=0.8)
        axes[2].add_patch(circle)
    
    axes[2].set_title(f'All Candidates (Red: Accepted, Blue: Rejected)', 
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Ground truth comparison (if available)
    if gt_coords is not None:
        axes[3].imshow(volume[:, :, slice_idx], cmap='gray')
        
        # Plot ground truth
        gt_in_slice = 0
        for coord in gt_coords:
            try:
                gx, gy, gz = int(float(coord[0])), int(float(coord[1])), int(float(coord[2]))
                if gz == slice_idx:
                    gt_in_slice += 1
                    rect = patches.Rectangle((gx - 3, gy - 3), 6, 6, 
                                           linewidth=2, edgecolor='lime', facecolor='none')
                    axes[3].add_patch(rect)
                    axes[3].text(gx + 4, gy - 4, f"GT{gt_in_slice}", 
                               color='lime', fontsize=10, weight='bold')
            except (ValueError, TypeError, IndexError):
                continue
        
        # Plot detections
        for candidate in final_in_slice:
            cx, cy, cz = candidate['coordinate']
            circle = plt.Circle((cy, cx), 4, color='red', fill=False, linewidth=2, alpha=0.7)
            axes[3].add_patch(circle)
        
        axes[3].set_title(f'GT vs Detections (GT: {gt_in_slice}, Pred: {len(final_in_slice)})', 
                         fontsize=14, fontweight='bold')
        axes[3].axis('off')
    
    plt.tight_layout()
    return fig

# ===============================================================
# 📚 PAGE CONTENT
# ===============================================================

# Home Page
if selected == "Home":
    st.markdown('<div class="section-headline">🧠 Deteksi Otomatis Perdarahan Mikro Serebral</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheadline">Sistem Cerdas Berbasis Deep Learning untuk Citra MRI</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📍 Apa Itu Proyek Ini?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="text-white">
    Proyek ini bertujuan untuk mengembangkan metode <b>otomatis</b> dalam mendeteksi <i>Cerebral Microbleeds (CMB)</i> – lesi kecil yang mencerminkan mikropendarahan di otak.<br><br>
    Lesi CMB penting dalam diagnosis karena berhubungan dengan:
    <ul>
    <li>🧠 Penurunan fungsi kognitif</li>
    <li>🧓 Risiko stroke & demensia</li>
    <li>❤️ Penyakit neurodegeneratif dan vaskular</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔬 Bagaimana Sistem Ini Bekerja?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="text-white">
    Sistem ini menggunakan pendekatan <b>Two-Stage Cascaded Framework</b>:<br><br>
    1. <b>3D FCN</b> untuk menyaring kandidat CMB awal<br>
    2. <b>3D CNN</b> untuk memastikan deteksi hanya pada lesi CMB asli<br>
    3. <b>Post-processing & evaluasi</b> untuk menyempurnakan hasil
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">⚙️ Langkah Penggunaan</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="text-white">
    1. Unggah citra MRI (SWI atau T2*-weighted)<br>
    2. Sistem akan melakukan pre-processing: <i>normalisasi, peningkatan kontras, brain masking</i>, dan <i>noise removal</i><br>
    3. Deteksi otomatis dilakukan<br>
    4. Hasil divisualisasikan secara interaktif
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎓 Tentang Kami</div>', unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown('<img src="https://i.imgur.com/ZPrugfv.jpg" class="circle-img">', unsafe_allow_html=True)
        with col2:
            st.markdown("**👩‍🔬 Benedicta Sabdaningtyas Pratita Pratanjana**  \nMahasiswa Teknik Biomedik ITS")

    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown('<img src="https://i.imgur.com/5WFquuD.jpg" class="circle-img">', unsafe_allow_html=True)
        with col2:
            st.markdown("**👨‍🏫 Dr. Norma Hermawan, S.T., M.T., M.Sc.**  \nDosen Pembimbing I")

    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown('<img src="https://i.imgur.com/sGTdZfZ.jpg" class="circle-img">', unsafe_allow_html=True)
        with col2:
            st.markdown("**👨‍🏫 Prof. Dr. Tri Arief Sardjono, S.T., M.T.**  \nDosen Pembimbing II")

# Get to Know Microbleeds Page
elif selected == "Get to Know Microbleeds":
    st.title("🧠 Get to Know Microbleeds")

    st.markdown("---")

    st.subheader("What are Cerebral Microbleeds?")
    st.write("""
    Cerebral microbleeds (CMBs) are small, round areas of chronic blood leakage in the brain. 
    They are often invisible in conventional MRI but can be detected using susceptibility-based sequences.
    While they are small in size, they may reflect more significant underlying vascular damage.
    """)

    st.markdown("---")

    st.subheader("🔍 Why Do They Matter?")
    st.write("""
    CMBs are clinically significant as they are associated with:
    - **Traumatic brain injury (TBI)**
    - **Stroke and cerebrovascular disease**
    - **Cognitive decline and dementia**
    
    Studies show that their presence may increase the risk of future hemorrhages and neurodegeneration.
    """)

    st.markdown("---")

    st.subheader("🧬 What Causes Them?")
    st.write("""
    CMBs often result from weakened or damaged small blood vessels. Common contributing factors include:
    - **Hypertension**
    - **Cerebral amyloid angiopathy**
    - **Aging**
    - **Head trauma**
    """)

    st.markdown("---")

    st.subheader("🧑‍⚕️ How Are They Detected?")
    st.write("""
    These microbleeds are typically found using:
    - **Susceptibility-Weighted Imaging (SWI)**
    - **Gradient-Recalled Echo (GRE) MRI**
    
    They appear as small, dark, round lesions—often located in deep or cortical brain regions.
    """)

    st.markdown("---")

    st.subheader("📸 Visual Example")
    st.image("https://www.ajnr.org/content/ajnr/32/6/1043/F1.large.jpg", caption="Source: AJNR, 2011")

    st.markdown("---")

    st.subheader("✨ Did You Know?")
    st.write("""
    Many individuals with CMBs are asymptomatic, and the microbleeds are only discovered incidentally 
    during imaging for unrelated conditions.
    """)

    st.markdown("---")

    st.subheader("📚 Learn More")
    st.write("""
    - [Cerebral Microbleeds: A Guide to Detection and Clinical Relevance (AJNR)](https://www.ajnr.org/content/32/6/1043)
    - [Cerebral Microbleeds: Imaging and Clinical Significance (RSNA)](https://pubs.rsna.org/doi/abs/10.1148/radiol.2018170803)
    - [Neuroimaging of CMBs (Radiopaedia)](https://radiopaedia.org/articles/cerebral-microbleeds)
    """)

    st.markdown("---")

    with st.expander("🎥 Optional: Watch a Lecture on CMBs"):
        st.write("""
        📺 Recommended segment: [2:36 to 17:00](https://www.youtube.com/watch?v=oSb_xKGhytY&t=156s)  
        This part covers how cerebral microbleeds appear on scans, their causes, and their clinical impact.
        """)
        st.video("https://www.youtube.com/watch?v=oSb_xKGhytY")

# Ground Truth Page
elif selected == "Ground Truth":
    st.title("🎯 Ground Truth Visualization")
    st.markdown("**Interactive CMB Ground Truth Explorer**")
    
    # Volume selection
    available_volumes = list(range(1, 21))  # Volumes 1-20
    
    st.markdown("### 📂 Dataset Selection")
    st.info("📊 Dataset: NII_MRI_CMB with 20 volumes (~315MB each)")
    
    # Selection mode
    selection_mode = st.radio(
        "Choose loading mode:",
        ["Single Volume (Recommended)", "Multiple Volumes", "Quick Demo"],
        help="Single volume recommended for faster loading (315MB download)"
    )
    
    selected_volumes = []
    
    if selection_mode == "Single Volume (Recommended)":
        selected_volume = st.selectbox("Select volume:", available_volumes, index=0)
        selected_volumes = [selected_volume]
        
    elif selection_mode == "Multiple Volumes":
        selected_volumes = st.multiselect(
            "Select volumes:", 
            available_volumes, 
            default=[1],
            max_selections=3,
            help="Maximum 3 volumes to prevent excessive loading (~945MB)"
        )
        
    else:  # Quick Demo
        st.info("Quick demo with pre-selected interesting volumes")
        demo_volumes = [1, 5, 10]  # Volumes with good examples
        selected_volumes = st.multiselect(
            "Demo volumes:", 
            demo_volumes, 
            default=[1],
            help="Pre-selected volumes with good CMB examples"
        )
    
    # Show download estimate
    if selected_volumes:
        total_size_mb = len(selected_volumes) * 315  # Actual size from HF
        st.info(f"📊 Estimated download: ~{total_size_mb} MB for {len(selected_volumes)} volume(s)")
        
        # Download warning for multiple volumes
        if len(selected_volumes) > 1:
            st.warning(f"⚠️ Loading {len(selected_volumes)} volumes will download ~{total_size_mb}MB. This may take several minutes.")
        
        # Load button
        if st.button("🚀 Load Selected Volumes", type="primary"):
            # Store loaded data in session state
            if 'loaded_cmb_volumes' not in st.session_state:
                st.session_state.loaded_cmb_volumes = {}
            
            # Load each volume
            for vol_id in selected_volumes:
                if vol_id not in st.session_state.loaded_cmb_volumes:
                    volume_data, gt_data = load_cmb_volume_data(vol_id)
                    
                    if volume_data is not None and gt_data is not None:
                        st.session_state.loaded_cmb_volumes[vol_id] = {
                            'volume': volume_data,
                            'gt': gt_data,
                            'shape': volume_data.shape
                        }
                        st.success(f"✅ Volume {vol_id} loaded: {volume_data.shape}")
                    else:
                        st.error(f"❌ Failed to load Volume {vol_id}")
    
    # Display loaded volumes
    if 'loaded_cmb_volumes' in st.session_state and st.session_state.loaded_cmb_volumes:
        st.markdown("### 🔍 CMB Volume Explorer")
        
        # Volume selector
        loaded_vol_ids = list(st.session_state.loaded_cmb_volumes.keys())
        current_vol = st.selectbox("Current volume:", loaded_vol_ids)
        
        vol_data = st.session_state.loaded_cmb_volumes[current_vol]
        volume = vol_data['volume']
        gt_data = vol_data['gt']
        
        # Extract ground truth coordinates (using 'cen' key)
        gt_coords = gt_data.get('cen', [])
        
        # Initialize slice index in session state
        slice_key = f'cmb_slice_idx_{current_vol}'
        if slice_key not in st.session_state:
            if len(gt_coords) > 0:
                st.session_state[slice_key] = int(gt_coords[0][2])
            else:
                st.session_state[slice_key] = volume.shape[2] // 2
        
        # CMB quick navigation
        if len(gt_coords) > 0:
            st.markdown("### 🎯 Quick CMB Navigation")
            cmb_slices = sorted(set(int(coord[2]) for coord in gt_coords))
            
            # Quick navigation buttons
            cols = st.columns(min(len(cmb_slices), 8))  # Max 8 buttons per row
            for i, slice_num in enumerate(cmb_slices[:8]):  # Show first 8 slices
                with cols[i % len(cols)]:
                    cmb_count_in_slice = sum(1 for coord in gt_coords if int(coord[2]) == slice_num)
                    if st.button(f"Slice {slice_num} ({cmb_count_in_slice})", key=f"nav_{current_vol}_{slice_num}"):
                        st.session_state[slice_key] = slice_num
                        st.rerun()
            
            if len(cmb_slices) > 8:
                st.info(f"+ {len(cmb_slices) - 8} more slices with CMBs")
        
        # Manual slice selector
        slice_idx = st.slider(
            "Manual slice selection:", 
            0, volume.shape[2] - 1, 
            st.session_state[slice_key],
            key=f"slice_slider_{current_vol}"
        )
        st.session_state[slice_key] = slice_idx
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(volume[:, :, slice_idx].T, cmap="gray", origin="lower")
        
        # Plot CMBs in current slice
        cmb_count = 0
        for i, coord in enumerate(gt_coords):
            try:
                x, y, z = int(float(coord[0])), int(float(coord[1])), int(float(coord[2]))
                if z == slice_idx:
                    cmb_count += 1
                    # Create rectangle around CMB
                    rect = patches.Rectangle((x - 3, y - 3), 6, 6, 
                                           linewidth=2, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x + 4, y - 4, f"CMB {cmb_count}", 
                           color='lime', fontsize=12, weight='bold')
            except (ValueError, TypeError, IndexError):
                continue
        
        ax.set_title(f'Volume {current_vol:02d} - Slice {slice_idx} - {cmb_count} CMBs visible', 
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        
        st.pyplot(fig)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Volume Shape", f"{volume.shape}")
        with col2:
            st.metric("Total CMBs", len(gt_coords))
        with col3:
            st.metric("CMBs in Slice", cmb_count)
        with col4:
            cmb_slices = len(set(int(coord[2]) for coord in gt_coords)) if gt_coords else 0
            st.metric("Slices with CMBs", cmb_slices)
        
        # Clear cache button
        if st.button("🗑️ Clear Loaded Volumes"):
            st.session_state.loaded_cmb_volumes = {}
            st.cache_data.clear()
            st.success("✅ Cache cleared!")
            st.rerun()
    
    else:
        st.warning("Please select at least one volume")

# CMB Detection Page
elif selected == "CMB Detection":
    st.title("🧠 CMB Detection Pipeline")
    st.markdown("**Two-Stage Cascaded Network: FCN + CNN**")
    
    st.markdown("""
    <div class="detection-card">
    <h3>🔄 Detection Workflow</h3>
    <ol>
    <li><strong>FCN Stage:</strong> 3D FCN processes full volume to generate candidate locations</li>
    <li><strong>CNN Stage:</strong> 3D CNN with SPP validates each candidate</li>
    <li><strong>Visualization:</strong> Interactive results with ground truth comparison</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Source Selection
    st.subheader("📂 Select Data Source")
    data_source = st.radio(
        "Choose your input method:",
        ["Use Pre-computed FCN Results", "Upload Custom FCN Results (.npy)", "Real-time FCN + CNN Inference"]
    )
    
    # ===============================================================
    # OPTION 1: Pre-computed FCN Results
    # ===============================================================
    if data_source == "Use Pre-computed FCN Results":
        st.markdown("### 🗂️ Pre-computed FCN Results")
        
        # Load FCN results
        if st.button("📥 Load Pre-computed FCN Results"):
            fcn_results = load_fcn_results()
            if fcn_results:
                st.session_state.precomputed_fcn_results = fcn_results
        
        if 'precomputed_fcn_results' in st.session_state:
            fcn_results = st.session_state.precomputed_fcn_results
            available_volumes = list(fcn_results.keys())
            
            # Volume selection options (Kaggle-style)
            st.markdown("### 🎯 Volume Selection (Kaggle-style)")
            
            selection_mode = st.selectbox(
                "Select processing mode:",
                ["Single Volume", "Multiple Volumes (Comma-separated)", "Volume Range", "All Volumes"]
            )
            
            selected_volumes = []
            
            if selection_mode == "Single Volume":
                selected_volume = st.selectbox("Choose volume:", available_volumes)
                selected_volumes = [selected_volume]
                
            elif selection_mode == "Multiple Volumes (Comma-separated)":
                vol_input = st.text_input(
                    "Enter volume IDs (e.g., 1,3,5,7):",
                    placeholder="1,3,5,7",
                    help=f"Available: {available_volumes}"
                )
                if vol_input:
                    try:
                        input_volumes = [int(x.strip()) for x in vol_input.split(",")]
                        selected_volumes = [v for v in input_volumes if v in available_volumes]
                        invalid = [v for v in input_volumes if v not in available_volumes]
                        if invalid:
                            st.warning(f"⚠️ Invalid volumes: {invalid}")
                        if selected_volumes:
                            st.success(f"✅ Selected: {selected_volumes}")
                    except ValueError:
                        st.error("❌ Use format: 1,3,5,7")
                        
            elif selection_mode == "Volume Range":
                col1, col2 = st.columns(2)
                with col1:
                    start_vol = st.selectbox("Start volume:", available_volumes, index=0)
                with col2:
                    end_vol = st.selectbox("End volume:", available_volumes, index=min(4, len(available_volumes)-1))
                
                start_idx = available_volumes.index(start_vol)
                end_idx = available_volumes.index(end_vol)
                if start_idx <= end_idx:
                    selected_volumes = available_volumes[start_idx:end_idx+1]
                else:
                    st.error("❌ Start must be <= End")
                    
            else:  # All Volumes
                selected_volumes = available_volumes.copy()
                st.warning(f"⚠️ Processing all {len(selected_volumes)} volumes")
            
            if selected_volumes:
                st.info(f"🎯 Selected volumes: {selected_volumes}")
                
                # Load CNN model
                cnn_model = load_cnn_model()
                
                if st.button("🚀 Start CNN Processing", type="primary"):
                    if cnn_model is None:
                        st.error("Please check CNN model loading!")
                    else:
                        # Process each selected volume
                        for vol_id in selected_volumes:
                            st.markdown(f"### 📊 Volume {vol_id} Results")
                            
                            # Get FCN data
                            fcn_data = fcn_results[vol_id]
                            fcn_candidates = fcn_data['fcn_candidates']
                            
                            # Load corresponding volume for CNN processing
                            volume_data, gt_data = load_cmb_volume_data(vol_id)
                            
                            if volume_data is not None:
                                # CNN Processing
                                X_patches, valid_candidates = extract_cnn_patches(volume_data, fcn_candidates)
                                
                                if X_patches is not None:
                                    # Real CNN inference
                                    final_candidates, rejected_candidates, cnn_probs = cnn_inference(
                                        cnn_model, X_patches, valid_candidates
                                    )
                                    
                                    # Results display
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("FCN Candidates", len(fcn_candidates))
                                    with col2:
                                        st.metric("Final Detections", len(final_candidates))
                                    with col3:
                                        precision = len(final_candidates) / len(fcn_candidates) if fcn_candidates else 0
                                        st.metric("Precision Ratio", f"{precision:.2f}")
                                    
                                    # Visualization with ground truth
                                    gt_coords = gt_data.get('cen', []) if gt_data else None
                                    
                                    if len(final_candidates) > 0:
                                        z_coords = [int(c['coordinate'][2]) for c in final_candidates]
                                        best_slice = max(set(z_coords), key=z_coords.count)
                                    else:
                                        best_slice = volume_data.shape[2] // 2
                                    
                                    fig = create_detection_visualization(
                                        volume_data, final_candidates, rejected_candidates, best_slice, gt_coords
                                    )
                                    st.pyplot(fig)
                                else:
                                    st.warning("No valid patches could be extracted from FCN candidates")
    
    # ===============================================================
    # OPTION 2: Upload Custom FCN Results
    # ===============================================================
    elif data_source == "Upload Custom FCN Results (.npy)":
        st.markdown("### 📤 Upload Custom FCN Results")
        
        uploaded_file = st.file_uploader(
            "Upload your FCN results (.npy file)",
            type=['npy'],
            help="Upload the .npy file containing FCN candidates from your Kaggle processing"
        )
        
        if uploaded_file is not None:
            # Auto-analyze uploaded .npy file
            if 'fcn_analysis' not in st.session_state or st.session_state.get('uploaded_filename') != uploaded_file.name:
                
                with st.spinner("🔍 Analyzing FCN results..."):
                    fcn_results, volume_info, processing_summary = analyze_fcn_npy(uploaded_file)
                
                if fcn_results is not None:
                    st.session_state.fcn_analysis = {
                        'fcn_results': fcn_results,
                        'volume_info': volume_info,
                        'processing_summary': processing_summary
                    }
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.success("✅ FCN results analyzed successfully!")
                else:
                    st.error("❌ Failed to analyze FCN results")
                    st.stop()
            
            # Display analysis results
            analysis = st.session_state.fcn_analysis
            
            # Summary metrics
            summary = analysis['processing_summary']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Volumes", summary['total_volumes'])
            with col2:
                st.metric("Total Candidates", summary['total_candidates'])
            with col3:
                st.metric("Volume Range", summary['volume_range'])
            with col4:
                avg_candidates = summary['avg_candidates_per_volume']
                st.metric("Avg Candidates/Volume", f"{avg_candidates:.1f}")
            
            # Detailed table
            st.markdown("### 📋 Volume Details")
            df = pd.DataFrame(analysis['volume_info'])
            st.dataframe(df, use_container_width=True)
            
            # Volume selection (Kaggle-style)
            available_volumes = summary['volume_ids']
            
            selection_mode = st.selectbox(
                "Select processing mode:",
                ["Single Volume", "Multiple Volumes (Comma-separated)", "Volume Range", "All Volumes"]
            )
            
            selected_volumes = []
            
            if selection_mode == "Single Volume":
                selected_volume = st.selectbox("Choose volume:", available_volumes)
                selected_volumes = [selected_volume]
                
            elif selection_mode == "Multiple Volumes (Comma-separated)":
                vol_input = st.text_input(
                    "Enter volume IDs (e.g., 1,3,5,7):",
                    placeholder="1,3,5,7",
                    help=f"Available: {available_volumes}"
                )
                if vol_input:
                    try:
                        input_volumes = [int(x.strip()) for x in vol_input.split(",")]
                        selected_volumes = [v for v in input_volumes if v in available_volumes]
                        invalid = [v for v in input_volumes if v not in available_volumes]
                        if invalid:
                            st.warning(f"⚠️ Invalid volumes: {invalid}")
                        if selected_volumes:
                            st.success(f"✅ Selected: {selected_volumes}")
                    except ValueError:
                        st.error("❌ Use format: 1,3,5,7")
                        
            elif selection_mode == "Volume Range":
                col1, col2 = st.columns(2)
                with col1:
                    start_vol = st.selectbox("Start volume:", available_volumes, index=0)
                with col2:
                    end_vol = st.selectbox("End volume:", available_volumes, index=min(4, len(available_volumes)-1))
                
                start_idx = available_volumes.index(start_vol)
                end_idx = available_volumes.index(end_vol)
                if start_idx <= end_idx:
                    selected_volumes = available_volumes[start_idx:end_idx+1]
                else:
                    st.error("❌ Start must be <= End")
                    
            else:  # All Volumes
                selected_volumes = available_volumes.copy()
                st.warning(f"⚠️ Processing all {len(selected_volumes)} volumes")
            
            if selected_volumes:
                st.info(f"🎯 Selected volumes: {selected_volumes}")
                
                # Extract selected FCN data
                selected_fcn_data = {vol_id: analysis['fcn_results'][vol_id] for vol_id in selected_volumes if vol_id in analysis['fcn_results']}
                
                # Store in session
                st.session_state.selected_fcn_data = selected_fcn_data
                st.session_state.selected_volumes = selected_volumes
                
                # CNN Processing
                if st.button("🔍 Process with CNN", type="primary"):
                    # Load CNN model
                    cnn_model = load_cnn_model()
                    
                    if cnn_model is None:
                        st.error("Failed to load CNN model!")
                    else:
                        for vol_id in selected_volumes:
                        if vol_id in selected_fcn_data:
                            st.markdown(f"### 📊 Volume {vol_id} CNN Results")
                            
                            fcn_data = selected_fcn_data[vol_id]
                            fcn_candidates = fcn_data['fcn_candidates']
                            
                            # Load corresponding volume
                            volume_data, gt_data = load_cmb_volume_data(vol_id)
                            
                            if volume_data is not None:
                                # CNN Processing
                                X_patches, valid_candidates = extract_cnn_patches(volume_data, fcn_candidates)
                                
                                if X_patches is not None:
                                    # Real CNN inference
                                    final_candidates, rejected_candidates, cnn_probs = cnn_inference(
                                        cnn_model, X_patches, valid_candidates
                                    )
                                    
                                    # Results display
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("FCN Candidates", len(fcn_candidates))
                                    with col2:
                                        st.metric("Final Detections", len(final_candidates))
                                    
                                    # Visualization
                                    gt_coords = gt_data.get('cen', []) if gt_data else None
                                    
                                    if len(final_candidates) > 0:
                                        z_coords = [int(c['coordinate'][2]) for c in final_candidates]
                                        best_slice = max(set(z_coords), key=z_coords.count)
                                    else:
                                        best_slice = volume_data.shape[2] // 2
                                    
                                    fig = create_detection_visualization(
                                        volume_data, final_candidates, rejected_candidates, best_slice, gt_coords
                                    )
                                    st.pyplot(fig)
                                else:
                                    st.warning("No valid patches could be extracted from FCN candidates")
    
    # ===============================================================
    # OPTION 3: Real-time FCN + CNN Inference
    # ===============================================================
    else:  # Real-time FCN + CNN Inference
        st.markdown("### ⚡ Real-time FCN + CNN Inference")
        st.info("🕐 Estimated processing time: ~45 minutes for full pipeline")
        
        # Configuration display
        st.markdown("#### ⚙️ Processing Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **FCN Settings:**
            - Patch size: {FCN_PATCH_SHAPE}
            - Stride: {FCN_STRIDE}
            - Batch size: {FCN_BATCH_SIZE}
            - Threshold: {FCN_THRESHOLD}
            """)
        with col2:
            st.markdown(f"""
            **CNN Settings:**
            - Patch size: {CNN_PATCH_SHAPE}
            - Threshold: {CNN_THRESHOLD}
            - Custom SPP layer: ✅
            """)
        
        # Volume selection for inference
        available_volumes = list(range(1, 21))
        selected_volume = st.selectbox("Select volume for inference:", available_volumes, index=0)
        
        # Model loading
        st.markdown("#### 🤖 Model Loading")
        
        # Option to use HF models or upload custom
        model_source = st.radio(
            "Choose model source:",
            ["Use HF Models (Recommended)", "Upload Custom Models"]
        )
        
        if model_source == "Use HF Models (Recommended)":
            st.info("📦 Models will be automatically loaded from Hugging Face")
            
            # Pre-load models
            if st.button("📥 Load Models from HF", type="primary"):
                fcn_model = load_fcn_model()
                cnn_model = load_cnn_model()
                
                if fcn_model and cnn_model:
                    st.session_state.fcn_model = fcn_model
                    st.session_state.cnn_model = cnn_model
                    st.success("✅ Both models loaded successfully!")
                else:
                    st.error("❌ Failed to load models")
            
            # Check if models are loaded
            models_ready = 'fcn_model' in st.session_state and 'cnn_model' in st.session_state
            
        else:  # Upload Custom Models
            fcn_model_file = st.file_uploader("Upload FCN Model (.h5)", type=['h5'], key="fcn_model")
            cnn_model_file = st.file_uploader("Upload CNN Model (.h5)", type=['h5'], key="cnn_model")
            models_ready = fcn_model_file and cnn_model_file
        
        if models_ready:
            # Start inference
            if st.button("🚀 Start Full Pipeline Inference", type="primary"):
                st.markdown(f"### 🔄 Processing Volume {selected_volume}")
                
                # Load volume
                volume_data, gt_data = load_cmb_volume_data(selected_volume)
                
                if volume_data is not None:
                    st.info(f"📊 Volume shape: {volume_data.shape}")
                    
                    # Get models
                    if model_source == "Use HF Models (Recommended)":
                        fcn_model = st.session_state.fcn_model
                        cnn_model = st.session_state.cnn_model
                    else:
                        # Load from uploaded files
                        with tempfile.NamedTemporaryFile(suffix='.h5') as fcn_temp:
                            fcn_temp.write(fcn_model_file.read())
                            fcn_temp.flush()
                            fcn_model = load_model(fcn_temp.name)
                            
                        with tempfile.NamedTemporaryFile(suffix='.h5') as cnn_temp:
                            cnn_temp.write(cnn_model_file.read())
                            cnn_temp.flush()
                            cnn_model = load_model(cnn_temp.name, custom_objects=CUSTOM_OBJECTS)
                    
                    # FCN Stage
                    st.markdown("#### 🎯 Stage 1: FCN Processing")
                    
                    # Create brain mask
                    brain_mask = create_brain_mask(volume_data)
                    st.success(f"✅ Brain mask created")
                    
                    # FCN inference
                    score_map = fcn_inference(fcn_model, volume_data)
                    score_map_masked = score_map * brain_mask
                    
                    # FCN clustering and NMS
                    fcn_candidates = fcn_clustering_nms(score_map_masked)
                    st.success(f"✅ FCN completed: {len(fcn_candidates)} candidates")
                    
                    # CNN Stage
                    st.markdown("#### 🎯 Stage 2: CNN Processing")
                    
                    X_patches, valid_candidates = extract_cnn_patches(volume_data, fcn_candidates)
                    
                    if X_patches is not None:
                        final_candidates, rejected_candidates, cnn_probs = cnn_inference(
                            cnn_model, X_patches, valid_candidates
                        )
                        
                        st.success(f"✅ CNN completed: {len(final_candidates)} final detections")
                        
                        # Results display
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("FCN Candidates", len(fcn_candidates))
                        with col2:
                            st.metric("Valid for CNN", len(valid_candidates))
                        with col3:
                            st.metric("Final Detections", len(final_candidates))
                        
                        # Visualization
                        gt_coords = gt_data.get('cen', []) if gt_data else None
                        
                        if len(final_candidates) > 0:
                            z_coords = [int(c['coordinate'][2]) for c in final_candidates]
                            best_slice = max(set(z_coords), key=z_coords.count)
                        else:
                            best_slice = volume_data.shape[2] // 2
                        
                        fig = create_detection_visualization(
                            volume_data, final_candidates, rejected_candidates, best_slice, gt_coords
                        )
                        st.pyplot(fig)
                        
                        # Detailed results
                        if final_candidates:
                            with st.expander(f"📋 Detailed Results - Volume {selected_volume}"):
                                for j, candidate in enumerate(final_candidates):
                                    coord = candidate['coordinate']
                                    st.write(f"**Detection {j+1}:** "
                                            f"Position ({coord[0]}, {coord[1]}, {coord[2]}) - "
                                            f"FCN Score: {candidate['fcn_score']:.3f}, "
                                            f"CNN Score: {candidate['cnn_score']:.3f}")
                    else:
                        st.warning("No valid patches could be extracted from FCN candidates")
                else:
                    st.error("Failed to load volume data")
        else:
            if model_source == "Use HF Models (Recommended)":
                st.warning("Please load models from Hugging Face first")
            else:
                st.warning("Please upload both FCN and CNN models to proceed")

# Chatbot Page
elif selected == "Chatbot":
    st.title("🧠 Cerebral Microbleeds Expert Chatbot")
    
    # Inisialisasi chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Hello! I'm Dr. Neuro, your AI specialist in cerebral microbleeds. How can I help you today?"
            }
        ]

    # Tampilkan chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input pengguna
    if prompt := st.chat_input("Ask about cerebral microbleeds..."):
        # Tambahkan ke history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Tampilkan pesan user
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate respons
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                from openai import OpenAI
                
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=st.secrets["OPENROUTER_API_KEY"],
                )

                # Format prompt untuk dokter spesialis
                system_prompt = """Anda adalah Dr. Neuro, ahli neurologi dan radiologi dengan spesialisasi cerebral microbleeds yang bisa menjawab dengan bahasa apa saja. 
                Berikan jawaban yang:
                - Singkat dan langsung (maksimal 3 paragraf)
                - Gunakan bahasa natural seperti sedang konsultasi
                - Fokus pada pertanyaan spesifik pasien
                - Sertakan poin-poin penting saja"""
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ]
                
                response = client.chat.completions.create(
                    model="deepseek/deepseek-r1-distill-llama-70b:free",
                    messages=messages,
                    temperature=0.7,
                    stream=True  # Untuk efek streaming
                )
                
                # Tampilkan respons secara bertahap
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                st.error(f"Maaf, terjadi error: {str(e)}")
                full_response = "Maaf, saya sedang tidak bisa menjawab. Silakan coba lagi nanti."
                message_placeholder.markdown(full_response)
        
        # Tambahkan ke history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Project FAQ Page
elif selected == "Project FAQ":
    st.markdown('<div class="section-title">❓ Frequently Asked Questions (FAQ)</div>', unsafe_allow_html=True)

    faqs = {
        "Apa itu Cerebral Microbleeds (CMB)?":
            "CMB adalah lesi kecil pada otak akibat kebocoran darah mikro, biasanya terlihat pada MRI jenis SWI atau GRE.",

        "Mengapa penting mendeteksi CMB?":
            "Karena CMB dapat menjadi indikator awal penyakit seperti stroke, demensia, atau gangguan vaskular lainnya.",

        "Jenis citra MRI apa yang digunakan dalam proyek ini?":
            "Sistem ini dirancang untuk mendeteksi CMB dari citra MRI jenis SWI (Susceptibility-Weighted Imaging) atau T2*-weighted GRE.",

        "Apa itu 3D FCN dan 3D CNN dalam konteks proyek ini?":
            "3D FCN digunakan untuk menyaring kandidat CMB awal dari seluruh volume otak, sedangkan 3D CNN bertugas untuk memverifikasi kandidat tersebut sebagai CMB sejati.",

        "Bisakah sistem ini digunakan untuk citra pasien asli?":
            "Iya, selama citra yang digunakan kompatibel (misalnya format .nii/.nii.gz dengan SWI/GRE), sistem ini dapat digunakan untuk membantu deteksi awal CMB.",

        "Apakah sistem ini menggantikan diagnosis dokter?":
            "Tidak. Sistem ini hanya alat bantu deteksi awal. Diagnosis akhir tetap harus dikonfirmasi oleh radiolog atau dokter spesialis saraf.",

        "Bagaimana cara kerja Two-Stage Cascaded Framework?":
            "Stage 1 (FCN) memproses seluruh volume MRI untuk menemukan kandidat CMB. Stage 2 (CNN dengan SPP layer) memvalidasi setiap kandidat untuk mengurangi false positive.",

        "Berapa lama waktu processing untuk satu volume?":
            "Untuk real-time inference: ~45 menit untuk full pipeline. Menggunakan pre-computed FCN results: ~2-5 menit untuk CNN stage saja.",

        "Apakah proyek ini open-source dan bisa dikembangkan?":
            "Ya, kode sumber proyek ini akan tersedia untuk umum dan bisa dikembangkan lebih lanjut untuk keperluan akademis dan riset."
    }

    for question, answer in faqs.items():
        with st.expander(f"❓ {question}"):
            st.markdown(f"<div class='text-white'>{answer}</div>", unsafe_allow_html=True)
