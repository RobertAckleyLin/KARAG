'''
dataset_preprocess.py
Convert robotic learning datasets(Open X-Embodiment) to 
        keyframe-keypoint format dataset 
        required for retrieval
pipeline:
    1. download Open-X-Emodiment dataset(https://robotics-transformer-x.github.io/) you want to use
    2. keyframe detection 
    (mainly following Paper RoboPrompt https://github.com/davidyyd/roboprompt)
    3. keypoint detection for each keyframe, using dinov2 features
    (mainly following Paper Rekep https://rekep-robot.github.io/)
    4. get keypoint trajectory using co-tracker
    (https://github.com/facebookresearch/co-tracker)
'''

import tensorflow_datasets as tfds
import tensorflow as tf

import numpy as np
from PIL import Image
from tqdm import tqdm

import os
import gc
# # add individual data points to replay
# def _add_keypoints_to_replay(
#         buffer,
#         i,
#         demo,
#         episode_keypoints,
#         epis_path_depth,
#         epis_path_char,
#         sim_name_to_real_name
#     ):
#     prev_action = None
#     cur_index = i

#     mask_dict = _get_mask_dict(epis_path_char, cur_index)

#     mask_id_to_sim_name_dict = _get_mask_id_to_name_dict(epis_path_char, cur_index)
#     point_cloud_dict = _get_point_cloud_dict(epis_path_depth, cur_index)
    
#     mask_id_to_sim_name = {}
#     for camera in CAMERAS:
#         mask_id_to_sim_name.update(mask_id_to_sim_name_dict[camera])

#     mask_id_to_real_name = {mask_id: sim_name_to_real_name[name] for mask_id, name in mask_id_to_sim_name.items()
#                         if name in sim_name_to_real_name}

#     avg_coord = form_obs(mask_dict, mask_id_to_real_name, point_cloud_dict)

#     buffer.append(avg_coord)
#     actions = []
#     for k, keypoint in enumerate(episode_keypoints):
#         obs_tp1 = demo[keypoint]
#         action = _get_action(
#             obs_tp1, obs_tp1)

#         actions.append(action)
    
#     buffer.append(actions)

# def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
#     next_is_not_final = i == (len(demo) - 2)
#     gripper_state_no_change = (
#             i < (len(demo) - 2) and
#             (obs.gripper_open == demo[i + 1].gripper_open and
#              obs.gripper_open == demo[i - 1].gripper_open and
#              demo[i - 2].gripper_open == demo[i - 1].gripper_open))
#     small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
#     stopped = (stopped_buffer <= 0 and small_delta and
#                (not next_is_not_final) and gripper_state_no_change)
#     return stopped

# def _keypoint_discovery(demo, delta=0.1) -> List[int]:
#     episode_keypoints = []
#     prev_gripper_open = demo[0].gripper_open
#     stopped_buffer = 0
#     for i, obs in enumerate(demo):
#         stopped = _is_stopped(demo, i, obs, stopped_buffer, delta)
#         stopped_buffer = 4 if stopped else stopped_buffer - 1
#         # if change in gripper, or end of episode.
#         last = i == (len(demo) - 1)
#         if i != 0 and (obs.gripper_open != prev_gripper_open or
#                         last or stopped):
#             episode_keypoints.append(i)
#         prev_gripper_open = obs.gripper_open
#     if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
#             episode_keypoints[-2]:
#         episode_keypoints.pop(-2)
#     #print('Found %d keypoints.' % len(episode_keypoints), episode_keypoints)
#     return episode_keypoints


display_key = 'image'
datasets_name = "viola"
batch_size = 8  # According to memory adjustment

PATH_TO_DATASET = f"/home/emma/ljz/KARAG/stage_1_Keypoint_Aware_Retrieval/dataset/{datasets_name}/0.1.0"
PATH_TO_OUTPUT_DIR = f'/home/emma/ljz/KARAG/stage_1_Keypoint_Aware_Retrieval/dataset/{datasets_name}/keyframes_keypoints'

dataset_builder = tfds.builder_from_directory(PATH_TO_DATASET)
# dataset_all = dataset_builder.as_dataset()
# dataset_all = dataset_builder.as_dataset(batch_size=)  # 使用默认缓冲区大小
dataset_all = dataset_builder.as_dataset(shuffle_files=False, read_config=tfds.ReadConfig(override_buffer_size=262144)) 



print("================================")
print("split of dataset : ", dataset_all.keys())

os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

instructions_file = open(os.path.join(PATH_TO_OUTPUT_DIR, 'instruction.txt'), 'a')
state_file = open(os.path.join(PATH_TO_OUTPUT_DIR, 'state.txt'), 'a')

# through the dataset
for ds_key in dataset_all.keys():
    ds = dataset_all[ds_key] 

    print(f"***** Processing {ds_key} split (batched) *****")
    for idx, episode in enumerate(ds):
        # make a folder for each video
        video_folder = os.path.join(PATH_TO_OUTPUT_DIR, f'episode_{idx}')
        os.makedirs(video_folder, exist_ok=True)
        # get all the frames of the episode
        frames = episode['steps']
        # through the frames
        state_list = []
        for frame_idx, step in tqdm(enumerate(frames)):
            # the feature key name of each dataset image are different, 
            # check the corresponding features.json file
            
            image = step['observation']["agentview_rgb"] # viola
            # image = step['observation'][image] # fractal20220817_data
            # image = step['observation']["image"] # bridge

            # natural_language_instruction = step["language_instruction"].numpy().decode('utf-8') # for ucsd、berkeley_fanuc_manipulation
            natural_language_instruction = step['observation']["natural_language_instruction"].numpy().decode('utf-8') 

            joint_states = step['observation']["joint_states"].numpy()
            gripper_states = step['observation']["gripper_states"].numpy()
            state_list.append(np.concatenate([joint_states, gripper_states]))

            # 将图像转换为 PIL 格式
            image_pil = Image.fromarray(image.numpy())

            # 保存图像，文件名格式为 frame_{frame_idx}.png
            output_path = os.path.join(video_folder, f'frame_{frame_idx}.png')
            image_pil.save(output_path)

            print("     ", natural_language_instruction)
            print("     ", joint_states)
            print("     ", gripper_states)

            print("     actions", step['action'])

            # 强制垃圾回收
            del image, image_pil
            if frame_idx % 50 == 0:  # 每50帧清理一次内存
                import gc
                gc.collect()

            import sys
            sys.exit(0)

        with open(state_file_path, 'a') as f:
            f.write(f"state {idx}: {state_list}\n")

        with open(instructions_file_path, 'a') as f:
            f.write(f"Video {idx} Instruction: {natural_language_instruction}\n")

            # 手动释放episode和frames的内存
        del episode, frames
        import gc
        gc.collect()

        print(f"第 {idx} 个视频的所有帧已保存到: {video_folder}, 该视频共有{frame_idx + 1}帧")

    print("所有视频的帧提取完成。")



