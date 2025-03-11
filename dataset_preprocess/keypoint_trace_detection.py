'''
keypoint_trace_detection.py
Convert robotic learning datasets(Open X-Embodiment) to 
        keypoint trace format dataset required for retrieval
pipeline:
    1. download Open-X-Emodiment dataset(https://robotics-transformer-x.github.io/) you want to use
    2. keypoint trace detection 
    (part of the code following Paper co-tracker(https://github.com/facebookresearch/co-tracke), 
                            and TraceVLA(https://github.com/umd-huang-lab/tracevla))
'''
import tensorflow_datasets as tfds
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # make sure TensorFlow do not use GPU

import os
import numpy as np
from PIL import Image

from key_trace_finder import KeyTraceFinder 
MODEL_PATH="/root/autodl-tmp/co-tracker/checkpoints/scaled_offline.pth"
key_trace_finder = KeyTraceFinder(
    cotracker_model_path=MODEL_PATH,
    device="cuda"
)

def process_episode(episode, episode_idx, datasets_name, ds_key, trace_idx):
    """Deal with a single episode"""
    frames = episode['steps']
    frames_list = list(frames.as_numpy_iterator())
    
    traces_info = []

    episode_keyframe_idxs = []
    prev_gripper = None
    for frame_idx, step in enumerate(frames):
        cur_gripper = int(step['action'][-1])
        if frame_idx==0 or cur_gripper!=prev_gripper or frame_idx==len(frames)-1:
            episode_keyframe_idxs.append(frame_idx)
        prev_gripper = cur_gripper

    for i in range(1, len(episode_keyframe_idxs)):
        begin_idx = episode_keyframe_idxs[i-1]
        end_idx = episode_keyframe_idxs[i]

        if end_idx - begin_idx < 5:
            continue
        
        gripper_state = int(frames_list[begin_idx]['action'][-1])
        if gripper_state < 0: # if open, continue
                continue
        
        image_sequence = []
        for step in frames_list[begin_idx:end_idx+1]:
            img_array = step['observation']["image"]
            img = Image.fromarray(img_array)
            image_sequence.append(img)

        key_trace = key_trace_finder.find_key_trace(image_sequence)
        
        print(f"key trace {key_trace.shape}")
        if key_trace.shape[0] < 5:
            continue
        # for test
        key_trace_finder.visualize_trace(
            image_sequence, 
            key_trace,
            output_path = f"demos/demo_trace{trace_idx}.gif",
            line_width = 3
        )
        
        natural_language_instruction = frames_list[begin_idx]["language_instruction"].decode('utf-8') 
        trace_info = {
            'trace_idx' : trace_idx,
            'trace_trace': key_trace,
            'gripper_state': gripper_state,
            'frame_begin_idx': begin_idx,
            'frame_end_idx': end_idx,
            'task_instruction': natural_language_instruction,
            'datasets_name' : datasets_name,
            'episode_id': f"{ds_key}_{episode_idx}",
        }
        traces_info.append(trace_info)
        trace_idx += 1
    
    return traces_info, trace_idx

# the feature key name of each dataset is different, check the corresponding features.json file
# datasets_name = "fractal"
# FRACTAL : https://www.tensorflow.org/datasets/catalog/fractal20220817_data
#       'steps': Dataset({
#            'action': FeaturesDict({
#               'gripper_closedness_action': Tensor(shape=(1,), dtype=float32, description=continuous gripper position),
#             }),
#           'observation': FeaturesDict({
#                'image': Image(shape=(256, 320, 3), dtype=uint8),
#                'natural_language_instruction': string,
#           }),
#       }),

# datasets_name = "viola"
# VIOLA : https://www.tensorflow.org/datasets/catalog/viola
#       'steps': Dataset({
#           'action': FeaturesDict({
#                'gripper_closedness_action': float32,
#            }),
#            'observation': FeaturesDict({
#               'agentview_rgb': Image(shape=(224, 224, 3), dtype=uint8, description=RGB captured by workspace camera),
#               'joint_states': Tensor(shape=(7,), dtype=float32, description=joint values),
#               'natural_language_instruction': string,
#       }),

datasets_name = "buds"
# BUDS : https://www.tensorflow.org/datasets/catalog/austin_buds_dataset_converted_externally_to_rlds
#     'steps': Dataset({
#         'action': Tensor(shape=(7,), dtype=float32, description=Robot action, consists of [6x end effector delta pose, 1x gripper position].),
#         'language_instruction': Text(shape=(), dtype=string),
#         'observation': FeaturesDict({
#             'image': Image(shape=(128, 128, 3), dtype=uint8, description=Main camera RGB observation.),
#             'state': Tensor(shape=(24,), dtype=float32, description=Robot state, consists of [7x robot joint angles, 1x gripper position, 16x robot end-effector homogeneous matrix].),
#     }),

PATH_TO_DATASET = f"/root/autodl-fs/open-x/{datasets_name}/0.1.0"
PATH_TO_OUTPUT_DIR = f'/root/autodl-fs/open-x/{datasets_name}/keypoint_trace'

os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)


dataset_builder = tfds.builder_from_directory(PATH_TO_DATASET)
dataset_all = dataset_builder.as_dataset()

all_traces = []
trace_idx = 0

for ds_key in dataset_all.keys():
    ds = dataset_all[ds_key] 

    print(f"Processing {ds_key} split...")
    for episode_idx, episode in enumerate(ds):
        traces, trace_idx = process_episode(episode, episode_idx, datasets_name, ds_key, trace_idx)
        all_traces.extend(traces)

        if episode_idx % 10 == 0:
            print(f"Processed {episode_idx} episode, get {len(all_traces)} traces")
                
np.savez_compressed(
    os.path.join(PATH_TO_OUTPUT_DIR, 'keypoint_traces.npz'),
    traces=all_traces
)
print(f"Total traces processed: {len(all_traces)}")