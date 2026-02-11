

import matplotlib.pyplot as plt
import torch
from cleanfid import fid
from metrics.cmmd_pytorch.main import compute_cmmd
from hashlib import md5
import torch.nn.functional as F
from transformers import ViTModel
from PIL import Image

from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

import os
import shutil
import glob
import torch
import numpy as np
import datetime
from natsort import natsorted
from IPython.display import clear_output
import datetime
import itertools
from pathlib import Path
from collections import defaultdict
from insightface.app import FaceAnalysis
import cv2
device = 'cuda:0'
torch.cuda.set_device(device)
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))



arcface_app = FaceAnalysis(det_name='buffalo_l')

arcface_app.prepare(ctx_id=0,det_thresh=0.05)  # Use GPU if available

def get_face_embedding(image_path):
    """Extract face embedding from an image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = arcface_app.get(img)


    # print(image_path,len(faces), "faces detected in the image")
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
        
    
    return faces[0].embedding

    # print(len(faces), "faces detected in the image:", image_path)
        
    # Draw boxes
    # for i,face in enumerate(faces):
    #     color = (0, 255, 0)
    #     if i > 0:
    #         color = (255, 0, 0)
    #     box = face.bbox.astype(int)
    #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
    # # Convert BGR to RGB for matplotlib display
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # Display using matplotlib
    # plt.figure(figsize=(6, 6))
    # plt.imshow(img_rgb)
    # plt.axis('off')
    # plt.title(f"Detected {len(faces)} face(s)")
    # plt.show()
    
    


def compare_faces(emb1, emb2, threshold=0.65):
    """
    Compare two embeddings using cosine similarity
    Args:
        emb1: First face embedding
        emb2: Second face embedding
        threshold: Decision threshold (default 0.65 is common for ArcFace)
    Returns:
        similarity_score: Cosine similarity value (0-1)
        is_same_person: Boolean indicating if similarity > threshold
    """
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity #, similarity > threshold


def get_image_files(directory):
    """Get all image files from a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    for file_path in Path(directory).rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    return image_files

def compute_arcface(generated_path, ref_path):
    """
    Compute ArcFace similarity scores between generated and reference images
    
    Args:
        generated_path: Directory containing generated images
        ref_path: Directory containing reference images
    
    Returns:
        average_similarity: Average cosine similarity score across all generated samples
        individual_scores: List of individual similarity scores for each generated image
        stats: Dictionary containing additional statistics
    """
    # Get all image files from both directories
    generated_images = get_image_files(generated_path)
    ref_images = get_image_files(ref_path)
    
    if not generated_images:
        raise ValueError(f"No images found in generated directory: {generated_path}")
    if not ref_images:
        raise ValueError(f"No images found in reference directory: {ref_path}")
    
    print(f"Found {len(generated_images)} generated images and {len(ref_images)} reference images")
    
    # Extract embeddings from all reference images
    ref_embeddings = []
    failed_ref = 0
    
    for ref_img in ref_images:
        try:
            embedding = get_face_embedding(ref_img)
            ref_embeddings.append(embedding)
        except Exception as e:
            print(f"Failed to process reference image {ref_img}: {e}")
            failed_ref += 1
    
    if not ref_embeddings:
        raise ValueError("No valid face embeddings extracted from reference images")
    
    # Compute average reference embedding
    ref_embeddings = np.array(ref_embeddings)
    avg_ref_embedding = np.mean(ref_embeddings, axis=0)
    
    # Normalize the average reference embedding
    avg_ref_embedding = avg_ref_embedding / np.linalg.norm(avg_ref_embedding)
    
    # Compute similarities for each generated image
    similarities = []
    failed_generated = 0
    
    for generated_img in generated_images:
        try:
            generated_embedding = get_face_embedding(generated_img)
            # Normalize the generated embedding
            generated_embedding = generated_embedding / np.linalg.norm(generated_embedding)
            
            # Compute cosine similarity with average reference embedding
            similarity = compare_faces(generated_embedding, avg_ref_embedding)
            similarities.append(similarity)
            
        except Exception as e:
            print(f"Failed to process generated image {generated_img}: {e}")
            failed_generated += 1
    
    if not similarities:
        raise ValueError("No valid similarities computed from generated images")
    
    # Compute final statistics
    average_similarity = np.mean(similarities)
    
    stats = {
        'num_generated_images': len(generated_images),
        'num_ref_images': len(ref_images),
        'successful_generated': len(similarities),
        'successful_ref': len(ref_embeddings),
        'failed_generated': failed_generated,
        'failed_ref': failed_ref,
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities),
        'std_similarity': np.std(similarities),
        'median_similarity': np.median(similarities)
    }
    
    print(f"\nResults:")
    print(f"Average similarity: {average_similarity:.4f}")
    print(f"Min similarity: {stats['min_similarity']:.4f}")
    print(f"Max similarity: {stats['max_similarity']:.4f}")
    print(f"Std similarity: {stats['std_similarity']:.4f}")
    print(f"Successful generated: {stats['successful_generated']}/{stats['num_generated_images']}")
    print(f"Successful references: {stats['successful_ref']}/{stats['num_ref_images']}")
    
    return average_similarity, similarities, stats




class DinoFacePipeline:
    def __init__(self, arcface_app,device='cuda:0'):
        """
        Initialize the DINO face pipeline
        
        Args:
            arcface_app: Pre-initialized ArcFace application for face detection
        """
        self.arcface_app = arcface_app
        
        # Load DINO ViT-S/16 model
        # self.dino_model = ViTModel.from_pretrained('facebook/dino-vits16')
        
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base')
        
        self.dino_model.to(device)
        self.dino_model.eval()
        
        # DINO transforms
        self.dino_transforms = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    def detect_and_crop_face(self, image_path, face_index=0):
        """
        Detect face in image and return cropped face
        
        Args:
            image_path (str): Path to input image
            face_index (int): Index of face to use if multiple faces detected
            
        Returns:
            PIL.Image: Cropped face image
            dict: Face detection info
        """
        # Read image using cv2 (BGR format) for face detection
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detect faces using BGR image
        faces = self.arcface_app.get(img_bgr)
        
        if len(faces) < 1:
            raise ValueError("No faces detected in the image")
        
        if len(faces) > 1:
            print(f"Warning: {len(faces)} faces detected. Using face at index {face_index}")
        
        if face_index >= len(faces):
            raise ValueError(f"Face index {face_index} out of range. Only {len(faces)} faces detected")
        
        # Get selected face bbox
        face = faces[face_index]
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Crop face from BGR image
        face_crop_bgr = img_bgr[y1:y2, x1:x2]
        
        # Convert BGR to RGB for PIL/DINO processing
        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_crop_rgb)
        
        face_info = {
            'bbox': bbox,
            'confidence': face.det_score,
            'face_index': face_index,
            'total_faces': len(faces),
            'face_crop_rgb': face_crop_rgb,
            'face_crop_pil': face_pil
        }
        
        return face_pil, face_info
    
    def get_dino_embedding(self, face_image):
        """
        Extract DINO embedding from face image
        
        Args:
            face_image (PIL.Image): Face image
            
        Returns:
            torch.Tensor: DINO embedding (CLS token)
        """
        # Apply DINO transforms
        face_tensor = self.dino_transforms(face_image).unsqueeze(0)  # Add batch dimension
        face_tensor = face_tensor.to(self.dino_model.device)  # Move to correct device
        # Get DINO features
        with torch.no_grad():
            outputs = self.dino_model(face_tensor)
        
        # Extract CLS token embedding
        last_hidden_states = outputs.last_hidden_state
        embedding = last_hidden_states[0, 0]  # First sample, CLS token
        
        return embedding
    
    def compute_face_similarity(self, image_path1, image_path2, face_index1=0, face_index2=0):
        """
        Compute similarity between faces in two images
        
        Args:
            image_path1 (str): Path to first image
            image_path2 (str): Path to second image
            face_index1 (int): Face index to use from first image
            face_index2 (int): Face index to use from second image
            
        Returns:
            dict: Results including similarity score and face info
        """
        # Detect and crop faces
        face1, face1_info = self.detect_and_crop_face(image_path1, face_index1)
        face2, face2_info = self.detect_and_crop_face(image_path2, face_index2)
        
        # Get DINO embeddings
        emb1 = self.get_dino_embedding(face1)
        emb2 = self.get_dino_embedding(face2)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(emb1, emb2, dim=0)
        
        results = {
            'similarity': similarity.item(),
            'face1_info': face1_info,
            'face2_info': face2_info,
            'face1_crop': face1,
            'face2_crop': face2,
            'embedding1': emb1,
            'embedding2': emb2
        }
        
        return results
    
    def get_image_files(self, directory):
        """
        Get all image files from a directory
        
        Args:
            directory (str): Path to directory
            
        Returns:
            list: List of image file paths
        """
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory, ext)))
            image_files.extend(glob.glob(os.path.join(directory, ext.upper())))
        
        return sorted(image_files)
    
    def compute_dino_face_similarity(self, generated_path, ref_path, compute_option='pairwise'):
        """
        Compute DINO face similarity scores between generated and reference images
        
        Args:
            generated_path (str): Directory containing generated images
            ref_path (str): Directory containing reference images
            compute_option (str): Method to compute similarity:
                'avg_ref' - compares against average reference embedding (default)
                'pairwise' - compares against each reference image individually
        
        Returns:
            tuple: (average_similarity, individual_scores, stats)
                - average_similarity: Average cosine similarity score
                - individual_scores: List of similarity scores (method depends on compute_option)
                - stats: Dictionary containing additional statistics
        """
        # Get all image files from both directories
        generated_images = self.get_image_files(generated_path)
        ref_images = self.get_image_files(ref_path)
        
        if not generated_images:
            raise ValueError(f"No images found in generated directory: {generated_path}")
        if not ref_images:
            raise ValueError(f"No images found in reference directory: {ref_path}")
        
        print(f"Found {len(generated_images)} generated images and {len(ref_images)} reference images")
        
        # Extract embeddings from all reference images
        ref_embeddings = []
        failed_ref = 0
        
        for ref_img in ref_images:
            try:
                face_image, _ = self.detect_and_crop_face(ref_img)
                embedding = self.get_dino_embedding(face_image)
                ref_embeddings.append(embedding)
            except Exception as e:
                print(f"Failed to process reference image {ref_img}: {e}")
                failed_ref += 1
        
        if not ref_embeddings:
            raise ValueError("No valid face embeddings extracted from reference images")
        
        # Normalize all reference embeddings
        ref_embeddings = [F.normalize(emb, p=2, dim=0) for emb in ref_embeddings]
        ref_embeddings = torch.stack(ref_embeddings)
        
        # Compute similarities for each generated image
        similarities = []
        failed_generated = 0
        pairwise_similarities = []  # Only used for 'pairwise' mode
        
        for generated_img in generated_images:
            try:
                face_image, face_info = self.detect_and_crop_face(generated_img)
                generated_embedding = self.get_dino_embedding(face_image)
                
                # print(generated_embedding.shape)
                generated_embedding = F.normalize(generated_embedding, p=2, dim=0)
                
                if compute_option == 'avg_ref':
                    # Current behavior - compare to average reference
                    avg_ref_embedding = torch.mean(ref_embeddings, dim=0)
                    avg_ref_embedding = F.normalize(avg_ref_embedding, p=2, dim=0)
                    similarity = F.cosine_similarity(generated_embedding, avg_ref_embedding, dim=0)
                    similarities.append(similarity.item())
                    
                elif compute_option == 'pairwise':
                    # New behavior - compare to each reference individually
                    pairwise_sims = F.cosine_similarity(generated_embedding.unsqueeze(0), 
                                                    ref_embeddings)
                    pairwise_similarities.append(pairwise_sims)
                    similarities.append(torch.mean(pairwise_sims).item())  # Average of all pairwise similarities
                    
                else:
                    raise ValueError(f"Invalid compute_option: {compute_option}. Must be 'avg_ref' or 'pairwise'")
                
                # display(face_image)
                print(f"similarity = {similarities[-1]:.4f}")
                # print(f"face_info['confidence']: {face_info['confidence']}")
                # print(f"Processed {generated_img}: ")

            except Exception as e:
                print(f"Failed to process generated image {generated_img}: {e}")
                failed_generated += 1
        
        if not similarities:
            raise ValueError("No valid similarities computed from generated images")
        
        # Convert similarities to tensor for easier computation
        similarities_tensor = torch.tensor(similarities)
        
        # Compute final statistics
        stats = {
            'num_generated_images': len(generated_images),
            'num_ref_images': len(ref_images),
            'successful_generated': len(similarities),
            'successful_ref': len(ref_embeddings),
            'failed_generated': failed_generated,
            'failed_ref': failed_ref,
            'min_similarity': torch.min(similarities_tensor).item(),
            'max_similarity': torch.max(similarities_tensor).item(),
            'std_similarity': torch.std(similarities_tensor).item(),
            'median_similarity': torch.median(similarities_tensor).item(),
            'compute_option': compute_option
        }
        
        if compute_option == 'pairwise':
            pairwise_similarities = torch.stack(pairwise_similarities)  # [num_generated, num_ref]
            stats.update({
                'pairwise_min': torch.min(pairwise_similarities).item(),
                'pairwise_max': torch.max(pairwise_similarities).item(),
                'pairwise_std': torch.std(pairwise_similarities).item()
            })
        
        print(f"\nResults ({compute_option}):")
        print(f"Average similarity: {torch.mean(similarities_tensor).item():.4f}")
        print(f"Min similarity: {stats['min_similarity']:.4f}")
        print(f"Max similarity: {stats['max_similarity']:.4f}")
        print(f"Std similarity: {stats['std_similarity']:.4f}")
        print(f"Successful generated: {stats['successful_generated']}/{stats['num_generated_images']}")
        print(f"Successful references: {stats['successful_ref']}/{stats['num_ref_images']}")
        
        if compute_option == 'pairwise':
            print(f"\nPairwise statistics:")
            print(f"All pairs min: {stats['pairwise_min']:.4f}")
            print(f"All pairs max: {stats['pairwise_max']:.4f}")
            print(f"All pairs std: {stats['pairwise_std']:.4f}")
        
        return torch.mean(similarities_tensor).item(), similarities, stats

dinofacepipeline = DinoFacePipeline(arcface_app,device=device)



def list_images_in_folder(folder_path, extensions={'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}):
    """
    Returns a list of image file paths from the given folder.
    
    Args:
        folder_path (str): Path to the folder.
        extensions (set): Set of allowed image file extensions (case-insensitive).

    Returns:
        List[str]: List of full paths to image files.
    """
    image_files = []
    for fname in os.listdir(folder_path):
        full_path = os.path.join(folder_path, fname)
        if os.path.isfile(full_path) and os.path.splitext(fname)[1].lower() in extensions:
            image_files.append(full_path)
    return image_files

def get_file_creation_date(filepath):
    timestamp = os.path.getctime(filepath)
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%d-%m-%y_%H:%M")

def calc_cdist_part(features_1, features_2, batch_size=10000):
    dists = []
    for feat2_batch in features_2.split(batch_size):
        dists.append(torch.cdist(features_1, feat2_batch).cpu())
    return torch.cat(dists, dim=1)


def calculate_precision_recall_part(features_1, features_2, neighborhood=3, batch_size=10000):
    # Precision
    dist_nn_1 = []
    for feat_1_batch in features_1.split(batch_size):
        dist_nn_1.append(calc_cdist_part(feat_1_batch, features_1, batch_size).kthvalue(neighborhood + 1).values)
    dist_nn_1 = torch.cat(dist_nn_1)
    precision = []
    for feat_2_batch in features_2.split(batch_size):
        dist_2_1_batch = calc_cdist_part(feat_2_batch, features_1, batch_size)
        precision.append((dist_2_1_batch <= dist_nn_1).any(dim=1).float())
    precision = torch.cat(precision).mean().item()
    # Recall
    dist_nn_2 = []
    for feat_2_batch in features_2.split(batch_size):
        dist_nn_2.append(calc_cdist_part(feat_2_batch, features_2, batch_size).kthvalue(neighborhood + 1).values)
    dist_nn_2 = torch.cat(dist_nn_2)
    recall = []
    for feat_1_batch in features_1.split(batch_size):
        dist_1_2_batch = calc_cdist_part(feat_1_batch, features_2, batch_size)
        recall.append((dist_1_2_batch <= dist_nn_2).any(dim=1).float())
    recall = torch.cat(recall).mean().item()
    return precision, recall


def get_features(base_dir, use_precompute_features_if_exist=False,max_count=-1):
    # TODO: right now, expect the features to be already exist only
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(base_dir, ext)))

    if not image_files:
        raise FileNotFoundError(f"No image files found in {base_dir}")

    first_image_path = natsorted(image_files)[0]
    date_str = get_file_creation_date(first_image_path)
        
    if use_precompute_features_if_exist:
        feature_dir = os.path.join(base_dir, "precomputed_features", "clip-l-14")
        # feature_dir = os.path.join(base_dir, "precomputed_features", "clip-b-32")
        feature_filename = f"{date_str}_n{max_count}.npy"
        feature_path = os.path.join(feature_dir, feature_filename)
        # feature_path = feature_path.replace("+", "--")  # Replace ':' with '-' for filename compatibility

        if os.path.exists(feature_path):
            print(f"Loading precomputed features from {feature_path}")
            feature = np.load(feature_path).astype("float32")
            feature = torch.tensor(feature)
            return feature
        else: print(f'{feature_path} does not exist')
            

def compute_pr(ref_path, eval_path, k=3, use_precompute_features_if_exist=False, feature=None, max_count=None):
    ref_features = get_features(ref_path, use_precompute_features_if_exist=use_precompute_features_if_exist, max_count=max_count)
    eval_features = get_features(eval_path, use_precompute_features_if_exist=use_precompute_features_if_exist, max_count=max_count)
    precision, recall = calculate_precision_recall_part(ref_features, eval_features, neighborhood=k)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1

def list_images_in_folder(folder_path, extensions={'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}):
    """
    Returns a list of image file paths from the given folder.
    
    Args:
        folder_path (str): Path to the folder.
        extensions (set): Set of allowed image file extensions (case-insensitive).

    Returns:
        List[str]: List of full paths to image files.
    """
    image_files = []
    for fname in os.listdir(folder_path):
        full_path = os.path.join(folder_path, fname)
        if os.path.isfile(full_path) and os.path.splitext(fname)[1].lower() in extensions:
            image_files.append(full_path)
    return image_files




def plot_multiple_score_curves(steps, all_scores, labels=None, title="Score vs Training Steps", save_path=None, method='KID'):
    plt.figure(figsize=(8, 5))
    for idx, scores in enumerate(all_scores):
        label = labels[idx] if labels else f"Exp {idx+1}"
        plt.plot(steps, scores, marker='o', linestyle='-', label=label)
    plt.xlabel("Training Step")
    plt.ylabel(f"{method.upper()} Score")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# obsolete
def create_limited_folder(original_path, limit, dummy_path):
    os.makedirs(dummy_path, exist_ok=True)
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(original_path, ext)))
    image_files = sorted(image_files)[:limit]
    for img_file in image_files:
        if os.path.isfile(img_file):
            shutil.copy(img_file, dummy_path)

def flatten_path(path):
    return os.path.normpath(path).replace(os.sep, '__')

def get_first_image_creation_date(folder):
    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp'):
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    if not image_files:
        raise FileNotFoundError(f"No images found in {folder}")
    first_img = natsorted(image_files)[0]
    ts = os.path.getctime(first_img)
    return datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%y_%H:%M")


def get_safe_filename(original_str):
    hash_id = md5(original_str.encode()).hexdigest()
    return hash_id

def build_score_cache_path(method, eval_path, ref_path, n_max_gen_img,apply_hash=True, other_params={}):
    f_eval = flatten_path(eval_path)
    f_ref  = flatten_path(ref_path)
    t_eval = get_first_image_creation_date(eval_path)
    t_ref  = get_first_image_creation_date(ref_path)
    filename = f"{f_eval}:{t_eval}--{f_ref}:{t_ref}_n{n_max_gen_img}"
    
    if method == 'pr' and "neighborhood" in other_params and other_params["neighborhood"] != 3:
        filename += f"_k{other_params['neighborhood']}"
    
    if apply_hash:
        # print('applying hash to filename')
        old_filename = filename
        filename = get_safe_filename(filename)
        # print(f"Old filename: {old_filename} -> New filename: {filename}")
    filename = f"{filename}.npy"
    return os.path.join("data_root", "cache", "precomputed_scores", method, filename)

def compute_distribution_score_multiexp(
    gen_img_paths,
    ref_img_path,
    steps,
    labels=None,
    device='cuda',
    n_max_gen_img=None,
    method='kid',
    use_precompute_features_if_exist=False,
    use_precompute_score_if_exist=False,
    clear_notebook_output=True,
    other_params={},
):
    if labels and len(labels) != len(gen_img_paths):
        raise ValueError("The number of labels must match the number of generated image paths.")

    all_scores = []

    for gen_path in gen_img_paths:
        scores = []
        for step in steps:
            path_for_eval = gen_path.format(step) if "{}" in gen_path else gen_path
            if n_max_gen_img: assert len(list_images_in_folder(path_for_eval)) >= n_max_gen_img
            
            cache_path = build_score_cache_path(method, path_for_eval, ref_img_path, n_max_gen_img,other_params=other_params)
            # cache_path = cache_path.replace("+", "--")  # Replace ':' with '-' for filename compatibility
            if use_precompute_score_if_exist:
                if os.path.exists(cache_path):
                    if method == 'pr':
                        score = np.load(cache_path)
                    else:
                        score = np.load(cache_path).item()
                    print(f"[CACHED] {method.upper()} from {cache_path}")
                else:
                    score = _compute(method, ref_img_path, path_for_eval, device, n_max_gen_img, use_precompute_features_if_exist, other_params=other_params)
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    np.save(cache_path, score)
                    print(f"[SAVED] {method.upper()} to {cache_path}")
            else:
                score = _compute(method, ref_img_path, path_for_eval, device, n_max_gen_img, use_precompute_features_if_exist, other_params=other_params)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, score)
                print(f"[SAVED] {method.upper()} to {cache_path}")
                
            if "{}" not in gen_path:
                print(f'{gen_path} | {method.upper()}: {score}')
                scores = [score] * len(steps)
                break
            else:
                print(f'{gen_path} | Step {step} - {method.upper()}: {score}')
                scores.append(score)

        all_scores.append(scores)

    # if clear_notebook_output: clear_output(wait=True)
    # if method == 'pr':
    #     Precisions = [[p for p, _, _ in sublist] for sublist in all_scores]
    #     Recalls = [[r for _, r, _ in sublist] for sublist in all_scores]
    #     F1s = [[f for _, _, f in sublist] for sublist in all_scores]
    #     plot_multiple_score_curves(steps, Precisions, labels, method='precision')
    #     plot_multiple_score_curves(steps, Recalls, labels, method='recall')
    #     plot_multiple_score_curves(steps, F1s, labels, method='f1')
    # else:
        
    #     plot_multiple_score_curves(steps, all_scores, labels, method=method)
    return all_scores

def _compute(method, ref_path, eval_path, device, n_max_gen_img, use_precompute_features_if_exist,other_params={}):
    if method == 'arcface':
        average_similarity, similarities, stats = compute_arcface(eval_path, ref_path)
        return average_similarity
    elif method == 'dinov2':
        average_similarity, similarities, stats = dinofacepipeline.compute_dino_face_similarity(eval_path, ref_path)
        return average_similarity
    elif method == 'kid':
        return fid.compute_kid(eval_path, ref_path, n_max_gen_img=n_max_gen_img, device=device, use_dataparallel=False)
    elif method == 'cmmd':
        return compute_cmmd(
            ref_path,
            eval_path,
            batch_size=10,
            max_count=-1 if n_max_gen_img is None else n_max_gen_img,
            use_precompute_features_if_exist=use_precompute_features_if_exist
        ).item()
    elif method == 'pr':
        print(f"k: {other_params.get('neighborhood', 3)}")
        return compute_pr(ref_path, eval_path, k=other_params.get('neighborhood', 3), max_count=n_max_gen_img,use_precompute_features_if_exist=use_precompute_features_if_exist)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    
    
    
    
    

# learn_concepts = ['osama','reese','honer','earle']
# rl_concepts = ['obama','rihanna','edsheeran','mrobbie']
learn_concepts = ['asante','reese','nivola','earle']
rl_concepts = ['obama','rihanna','edsheeran','mrobbie']

concepts = rl_concepts + learn_concepts
base_cfgs = [7.5, 6.0]
base_cfgs = [7.5]
training_steps = list(range(0, 1001, 100))
seeds = [0,1,2]

for concept in concepts:
    for base_cfg in base_cfgs: 
        for seed in seeds:
            gen_img_paths = []; labels = []
            seed_tag = '' if seed == 0 else f'.r{seed}'
            if concept in learn_concepts:
                gen_img_paths +=  [f'data_root/generated/model/ch.ct.l4.kv_{concept}A5V0-V_pr1.00.neg_ln.lr1e-4.ti5e-4_b1g4{seed_tag}/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}']
                labels += [f"learn_{concept}_{seed}"]
            if concept in rl_concepts:
                gen_img_paths += [f'data_root/generated/model/rlct4.reV.{concept}A5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4{seed_tag}_ul1.prg1e-4d8e+3.lr1e-4.n8.G.{concept}.person.s50{seed_tag}_rv/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}']
                labels += [f"relearn_{concept}_{seed}"]
            ref_img_path = f'data_root/data/real_data/{concept}/aligned/{concept}-5-v0/'   
            all_scores = compute_distribution_score_multiexp(gen_img_paths, ref_img_path, training_steps, labels, device=device, n_max_gen_img=5, method='dinov2',use_precompute_features_if_exist=True,use_precompute_score_if_exist=True,clear_notebook_output=False)






learn_concepts = ['asante','reese','nivola','earle']
rl_concepts = ['obama','rihanna','edsheeran','mrobbie']

concepts = rl_concepts + learn_concepts
base_cfgs = [7.5, 6.0]
training_steps = list(range(0, 1001, 50))
seeds = [0,1,2]

for concept in concepts:
    for base_cfg in base_cfgs: 
        for seed in seeds:
            gen_img_paths = []; labels = []
            seed_tag = '' if seed == 0 else f'.r{seed}'
            if concept in learn_concepts:
                gen_img_paths +=  [f'data_root/generated/model/ch.ct.l4.kv_{concept}A5V0-V_pr1.00.neg_ln.lr1e-4.ti5e-4_b1g4{seed_tag}/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}']
                labels += [f"learn_{concept}_{seed}"]
            if concept in rl_concepts:
                gen_img_paths += [f'data_root/generated/model/rlct4.reV.{concept}A5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4{seed_tag}_ul1.prg1e-4d8e+3.lr1e-4.n8.G.{concept}.person.s50{seed_tag}_rv/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}']
                labels += [f"relearn_{concept}_{seed}"]
            ref_img_path = f'data_root/data/real_data/{concept}/aligned/{concept}-5-v0/'   
            all_scores = compute_distribution_score_multiexp(gen_img_paths, ref_img_path, training_steps, labels, device=device, n_max_gen_img=5, method='dinov2',use_precompute_features_if_exist=True,use_precompute_score_if_exist=True,clear_notebook_output=False)







learn_concepts = ['asante','reese','nivola','earle']
rl_concepts = ['obama','rihanna','edsheeran','mrobbie']

concepts = rl_concepts + learn_concepts
base_cfgs = [7.5, 6.0]
training_steps = list(range(0, 1001, 100))
seeds = [3,4]

for concept in concepts:
    for base_cfg in base_cfgs: 
        for seed in seeds:
            gen_img_paths = []; labels = []
            seed_tag = '' if seed == 0 else f'.r{seed}'
            if concept in learn_concepts:
                gen_img_paths +=  [f'data_root/generated/model/ch.ct.l4.kv_{concept}A5V0-V_pr1.00.neg_ln.lr1e-4.ti5e-4_b1g4{seed_tag}/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}']
                labels += [f"learn_{concept}_{seed}"]
            if concept in rl_concepts:
                gen_img_paths += [f'data_root/generated/model/rlct4.reV.{concept}A5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4{seed_tag}_ul1.prg1e-4d8e+3.lr1e-4.n8.G.{concept}.person.s50{seed_tag}_rv/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}']
                labels += [f"relearn_{concept}_{seed}"]
            ref_img_path = f'data_root/data/real_data/{concept}/aligned/{concept}-5-v0/'   
            all_scores = compute_distribution_score_multiexp(gen_img_paths, ref_img_path, training_steps, labels, device=device, n_max_gen_img=5, method='dinov2',use_precompute_features_if_exist=True,use_precompute_score_if_exist=True,clear_notebook_output=False)




learn_concepts = ['asante','reese','nivola','earle']
rl_concepts = ['obama','rihanna','edsheeran','mrobbie']

concepts = rl_concepts + learn_concepts
base_cfgs = [7.5, 6.0]
training_steps = list(range(50, 1001, 100))
seeds = [0,1,2]

for concept in concepts:
    for base_cfg in base_cfgs: 
        for seed in seeds:
            gen_img_paths = []; labels = []
            seed_tag = '' if seed == 0 else f'.r{seed}'
            if concept in learn_concepts:
                gen_img_paths +=  [f'data_root/generated/model/ch.ct.l4.kv_{concept}A5V0-V_pr1.00.neg_ln.lr1e-4.ti5e-4_b1g4{seed_tag}/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}']
                labels += [f"learn_{concept}_{seed}"]
            if concept in rl_concepts:
                gen_img_paths += [f'data_root/generated/model/rlct4.reV.{concept}A5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4{seed_tag}_ul1.prg1e-4d8e+3.lr1e-4.n8.G.{concept}.person.s50{seed_tag}_rv/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}']
                labels += [f"relearn_{concept}_{seed}"]
            ref_img_path = f'data_root/data/real_data/{concept}/aligned/{concept}-5-v0/'   
            all_scores = compute_distribution_score_multiexp(gen_img_paths, ref_img_path, training_steps, labels, device=device, n_max_gen_img=5, method='arcface',use_precompute_features_if_exist=True,use_precompute_score_if_exist=True)





concepts = rl_concepts + learn_concepts
base_cfgs = [7.5, 6.0]
training_steps = list(range(0, 1001, 100))
seeds = [3,4]

for concept in concepts:
    for base_cfg in base_cfgs: 
        for seed in seeds:
            gen_img_paths = []; labels = []
            seed_tag = '' if seed == 0 else f'.r{seed}'
            if concept in learn_concepts:
                gen_img_paths +=  [f'data_root/generated/model/ch.ct.l4.kv_{concept}A5V0-V_pr1.00.neg_ln.lr1e-4.ti5e-4_b1g4{seed_tag}/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}']
                labels += [f"learn_{concept}_{seed}"]
            if concept in rl_concepts:
                gen_img_paths += [f'data_root/generated/model/rlct4.reV.{concept}A5V0.ln.lr1e-4.ti5e-4.pr1.00.neg.b1g4{seed_tag}_ul1.prg1e-4d8e+3.lr1e-4.n8.G.{concept}.person.s50{seed_tag}_rv/checkpoint-{{}}/a photo of v1_neg/{base_cfg:.2f}']
                labels += [f"relearn_{concept}_{seed}"]
            ref_img_path = f'data_root/data/real_data/{concept}/aligned/{concept}-5-v0/'   
            all_scores = compute_distribution_score_multiexp(gen_img_paths, ref_img_path, training_steps, labels, device=device, n_max_gen_img=5, method='arcface',use_precompute_features_if_exist=True,use_precompute_score_if_exist=True)



