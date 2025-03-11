'''
keyframe_detection.py
Convert robotic learning datasets(Open X-Embodiment) to 
        keyframe format dataset required for retrieval
pipeline:
    1. download Open-X-Emodiment dataset(https://robotics-transformer-x.github.io/) you want to use
    2. keyframe detection 
    (mainly following Paper RoboPrompt https://github.com/davidyyd/roboprompt)
'''

import tensorflow_datasets as tfds

import os
import numpy as np
from PIL import Image

def filter_adjacent_idxs(idxs):
    """Remove adjacent idxs(continuous idxs with a difference<=1), and retain idxs with larger intervals"""
    if len(idxs) <= 1:
        return idxs.copy()
    
    filtered = [idxs[0]]
    for k_id in idxs[1:]:
        if k_id - filtered[-1] > 1:
            filtered.append(k_id)
    return filtered

def save_keyframes(frames, keyframe_idxs, keyframe_folder):
    """save the keyframes pics for visualization"""
    for frame_idx, step in enumerate(frames):
        if frame_idx in keyframe_idxs:
            # save the image
            image_pil = Image.fromarray(step['observation']["image"].numpy())
            gripper_states = int(step['action'].numpy()[-1])
            output_path = os.path.join(keyframe_folder, f'frame_{frame_idx}_{gripper_states}.png')
            image_pil.save(output_path)

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
#             'joint_states': Tensor(shape=(7,), dtype=float32, description=joint values),
#             'natural_language_instruction': string,
#       }),

datasets_name = "buds"
# BUDS : https://www.tensorflow.org/datasets/catalog/austin_buds_dataset_converted_externally_to_rlds
#     'steps': Dataset({
#         'action': Tensor(shape=(7,), dtype=float32, description=Robot action, consists of [6x end effector delta pose, 1x gripper position].),
#         'language_instruction': Text(shape=(), dtype=string),
#         'observation': FeaturesDict({
#             'image': Image(shape=(128, 128, 3), dtype=uint8, description=Main camera RGB observation.),
#             'state': Tensor(shape=(24,), dtype=float32, description=Robot state, consists of [7x robot joint angles, 1x gripper position, 16x robot end-effector homogeneous matrix].),
#         }),

PATH_TO_DATASET = f"PATH_TO_YOUR_DATASET/{datasets_name}/0.1.0"
PATH_TO_OUTPUT_DIR = f'PATH_TO_YOUR_OUTPUT_DIR/{datasets_name}/keyframes'

# Keyframe detection parameters
DELTA = 0.0001  # Joint speed threshold
STOP_BUFFER = 4  # Stop buffer size

dataset_builder = tfds.builder_from_directory(PATH_TO_DATASET)
dataset_all = dataset_builder.as_dataset()
print("================================")
print("split of dataset : ", dataset_all.keys())

os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)
instructions_file = open(os.path.join(PATH_TO_OUTPUT_DIR, 'instruction.txt'), 'a')
keyframes_file = open(os.path.join(PATH_TO_OUTPUT_DIR, 'keyframes.txt'), 'a')

# through the dataset
total_frames = 0
total_keyframes = 0

for ds_key in dataset_all.keys():
    ds = dataset_all[ds_key] 

    print(f"***** Processing {ds_key} split (batched) *****")
    for idx, episode in enumerate(ds):
        # get all the frames of the episode
        frames = episode['steps']

        # variables used for keyframe detection 
        episode_keyframe_idxs = [0, ]

        prev_joint = None
        prev_gripper = None
        prev_prev_gripper = None
        stopped_buffer = 0

        stopped_by_gripper = 0
        stopped_by_vel = 0

        for frame_idx, step in enumerate(frames):
            if frame_idx == 0:
                prev_joint = step['observation']['state'].numpy()[:7]
                continue

            natural_language_instruction = step["language_instruction"].numpy().decode('utf-8') 
            cur_gripper = step['action'].numpy()[-1]
            cur_joint = step['observation']['state'].numpy()[:7]

            gripper_consistent = (cur_gripper == prev_gripper) if prev_gripper is not None else False
            
            small_vel = np.allclose(cur_joint-prev_joint, 0, atol=DELTA)
            next_is_not_final = frame_idx < (len(frames) - 1)            
            # update stopped_buffer
            if stopped_buffer <= 0 and small_vel and gripper_consistent and next_is_not_final:
                if frame_idx >= 2 and (cur_gripper == prev_prev_gripper):
                    stopped = True
                    stopped_buffer = STOP_BUFFER
                else:
                    stopped = False
            else:
                stopped = False
                stopped_buffer = max(stopped_buffer - 1, 0)
            
            # gripper state change, or last frame, or stop state
            if (cur_gripper != prev_gripper) or (frame_idx == len(frames)-1) or stopped:
                episode_keyframe_idxs.append(frame_idx)

                if cur_gripper != prev_gripper:
                    stopped_by_gripper += 1 
                elif stopped:
                    stopped_by_vel += 1

            # update
            prev_prev_gripper = prev_gripper
            prev_gripper = cur_gripper
            prev_joint = cur_joint

        # Post processing keyframes (removing adjacent keyframes)
        filtered_episode_keyframe_idxs = filter_adjacent_idxs(episode_keyframe_idxs)
        total_frames += len(frames)
        total_keyframes += len(filtered_episode_keyframe_idxs)

        keyframes_file.write(f"{ds_key} Episode {idx} Keyframe Idxs: {','.join(map(str, filtered_episode_keyframe_idxs))}\n")
        instructions_file.write(f"{ds_key} Episode {idx} Instruction: {natural_language_instruction}\n")

        print(f"     {idx} th episode, convert {len(frames)} frames to {len(filtered_episode_keyframe_idxs)} keyframes")
        print(f"                     stopped_by_gripper = {stopped_by_gripper}, stopped_by_vel = {stopped_by_vel}")
        print(f"                     {filtered_episode_keyframe_idxs}")
        
        ## for visualization, 
        # make a folder for each eposide
        keyframe_folder = os.path.join(PATH_TO_OUTPUT_DIR, f'episode_{ds_key}_{idx}')
        os.makedirs(keyframe_folder, exist_ok=True)
        save_keyframes(frames, filtered_episode_keyframe_idxs, keyframe_folder)
    print("Done !")

keyframes_file.close()
instructions_file.close()
print(f"total_frames = {total_frames}, total_keyframes = {total_keyframes} -----> {total_keyframes/total_frames}")