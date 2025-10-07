import os
import sys
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import shutil

tpsmm_path = os.path.abspath("Thin-Plate-Spline-Motion-Model")
if tpsmm_path not in sys.path:
    sys.path.append(tpsmm_path)

from demo import load_checkpoints, make_animation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ AI Pipeline: Using device: {DEVICE}")

EFFICIENTNET_PATH = "checkpoints/efficientnet_b3_body_pose_best.pth"
VIT_PATH = "checkpoints/vit_finetuned_skeleton.pth"
TPSMM_PATH = "checkpoints/taichi.pth.tar"
TPSMM_CONFIG_PATH = "taichi-256.yaml"

DRIVING_VIDEOS = {
    "Taichi 1": "driving_videos/taichi5.mp4",
    "Taichi 2": "driving_videos/taichi6.mp4",
    "Taichi 3": "driving_videos/taichi7.mp4",
    "Taichi 4": "driving_videos/taichi8.mp4",
}

mp_pose = mp.solutions.pose
POSE_DETECTOR = mp_pose.Pose(static_image_mode=True, model_complexity=1)

REQUIRED_KEYPOINTS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,     mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,       mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,      mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,     mp_pose.PoseLandmark.RIGHT_ANKLE,
]
VISIBILITY_THRESHOLD = 0.6

def create_classifier_model(model_name, num_ftrs_layer_name, checkpoint_path):
    model = timm.create_model(model_name, pretrained=False)
    if 'vit' in model_name:
        num_ftrs = getattr(model, num_ftrs_layer_name).in_features
        model.head = nn.Sequential(nn.Linear(num_ftrs, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid())
    else:
        num_ftrs = getattr(model, num_ftrs_layer_name).in_features
        model.classifier = nn.Sequential(nn.Linear(num_ftrs, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid())
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded {model_name} model from {checkpoint_path}")
    return model

def load_tpsmm_model():
    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path=TPSMM_CONFIG_PATH, checkpoint_path=TPSMM_PATH, device=DEVICE)
    print(f"Loaded TPSMM model from {TPSMM_PATH}")
    return inpainting, kp_detector, dense_motion_network, avd_network

EFFICIENTNET_MODEL = create_classifier_model('efficientnet_b3', 'classifier', EFFICIENTNET_PATH)
VIT_MODEL = create_classifier_model('vit_base_patch16_224', 'head', VIT_PATH)
TPSMM_MODELS = load_tpsmm_model()
print("All AI models loaded and ready.")

def run_efficientnet_b3(image_path: str) -> bool:
    print(f"Running EfficientNetB3 check on {image_path}...")
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        img_transforms = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image_tensor = img_transforms(image).unsqueeze(0).to(DEVICE)
        output = EFFICIENTNET_MODEL(image_tensor).squeeze()
        prediction = (output >= 0.5).item()
        return not bool(prediction)

def generate_and_check_skeleton(original_image_path: str, job_id: str) -> (bool, str or None):
    print(f"Generating and checking skeleton for {original_image_path}...")
    
    frame = cv2.imread(original_image_path)
    if frame is None:
        return False, None
        
    results = POSE_DETECTOR.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if not results.pose_landmarks:
        print("Skeleton Check Failed: No pose landmarks detected.")
        return False, None

    for landmark_enum in REQUIRED_KEYPOINTS:
        visibility = results.pose_landmarks.landmark[landmark_enum.value].visibility
        if visibility < VISIBILITY_THRESHOLD:
            print(f"Skeleton Check Failed: A key joint ({landmark_enum.name}) was not visible enough.")
            return False, None

    skeleton_image = np.zeros_like(frame)
    mp.solutions.drawing_utils.draw_landmarks(
        image=skeleton_image,
        landmark_list=results.pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255,255,255), thickness=2),
        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255,255,255), thickness=2)
    )
    
    skeleton_filename = f"{job_id}_skeleton.jpg"
    skeleton_path = os.path.join("static/uploads", skeleton_filename)
    cv2.imwrite(skeleton_path, skeleton_image)
    
    print(f"Skeleton check passed. Image saved to {skeleton_path}")
    return True, skeleton_path

def run_vit_skeleton(skeleton_image_path: str) -> bool:
    print(f"Running ViT model validation on {skeleton_image_path}...")
    with torch.no_grad():
        image = Image.open(skeleton_image_path).convert("RGB")
        vit_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        image_tensor = vit_transform(image).unsqueeze(0).to(DEVICE)
        output = VIT_MODEL(image_tensor).squeeze()
        prediction = (output >= 0.5).item()
        return bool(prediction)

def run_tpsmm_animation(job_id: str, image_path: str, motion_id: str) -> str:
    print(f"Running REAL TPSMM with motion '{motion_id}' on {image_path}...")
    
    driving_video_path = DRIVING_VIDEOS.get(motion_id)
    if not driving_video_path or not os.path.exists(driving_video_path):
        raise ValueError(f"Invalid motion_id or driving video file not found: '{motion_id}'")
        
    pixel = 256
    
    source_image = imageio.imread(image_path)
    reader = imageio.get_reader(driving_video_path)
    fps = reader.get_meta_data()['fps']

    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    
    source_image_resized = resize(source_image, (pixel, pixel))[..., :3]
    driving_video_resized = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]

    inpainting, kp_detector, dense_motion_network, avd_network = TPSMM_MODELS

    print("Generating animation... This may take a few minutes.")
    predictions = make_animation(
        source_image_resized, 
        driving_video_resized, 
        inpainting, 
        kp_detector, 
        dense_motion_network, 
        avd_network, 
        device=DEVICE, 
        mode='relative'
    )
    print("Animation generation complete.")
    
    output_filename = f"{job_id}.mp4"
    output_path = os.path.join("static/results", output_filename)
    
    print(f"Saving video to file: {output_path}")
    imageio.mimsave(output_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    print(f"TPSMM: Video successfully saved.")

    return output_filename


def run_full_pipeline(job_id, image_path, motion_id, JOBS_dict):
    skeleton_image_path = None
    try:
        JOBS_dict[job_id]['step'] = 'Step 1/4: Analyzing body pose...'
        if not run_efficientnet_b3(image_path):
            raise Exception("Image is not a full-body pose. Please use a different photo.")

        JOBS_dict[job_id]['step'] = 'Step 2/4: Generating and checking skeleton...'
        is_good_pose, skeleton_image_path = generate_and_check_skeleton(image_path, job_id)
        if not is_good_pose:
            raise Exception("Key body parts are not visible. Please use a clearer photo.")

        JOBS_dict[job_id]['step'] = 'Step 3/4: Validating skeleton structure...'
        if not run_vit_skeleton(skeleton_image_path):
            raise Exception("Skeleton structure is not suitable for animation.")
        
        JOBS_dict[job_id]['step'] = 'Step 4/4: Generating video (this can take a minute)...'
        result_filename = run_tpsmm_animation(job_id, image_path, motion_id)
        
        JOBS_dict[job_id]['status'] = 'completed'
        JOBS_dict[job_id]['result_url'] = f"/api/results/{result_filename}"
        print(f"Job {job_id}: Completed successfully!")

    except Exception as e:
        print(f"Job {job_id}: FAILED. Reason: {e}")
        JOBS_dict[job_id]['status'] = 'error'
        JOBS_dict[job_id]['message'] = str(e)
    finally:
        if skeleton_image_path and os.path.exists(skeleton_image_path):
            os.remove(skeleton_image_path)
            print(f"Cleaned up temporary file: {skeleton_image_path}")