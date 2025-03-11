
'''
For each item of data in the dataset
    calculate the RB-fastDTW similarity between the input trace and the trace of the data item 
Return the K data items with the highest similarity
(   
    The code implementation here also incorporates CLIP semantic similarity, 
    but we found that it is not very meaningful after adding it. 
    Therefore, setting alpha to 0 is sufficient
)
'''

import numpy as np
import torch

from fastdtw import fastdtw
from heapq import nsmallest
from scipy.spatial.distance import cosine
from transformers import CLIPTextModel, CLIPTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
clip_model = CLIPTextModel.from_pretrained(model_name).to(device)

def uniform_sample(trace, num_points=20):
    indices = np.linspace(0, len(trace)-1, num_points, dtype=int)
    return trace[indices]

def min_max_normalize(series):
    return (series - np.min(series)) / (np.max(series) - np.min(series))

def normalize_trajectory(trajectory):
    x_coords = np.array([p[0] for p in trajectory])
    y_coords = np.array([p[1] for p in trajectory])

    x_normalized = min_max_normalize(x_coords)
    y_normalized = min_max_normalize(y_coords)

    normalized_trajectory = list(zip(x_normalized, y_normalized))
    
    return normalized_trajectory

def rotate_trajectory(traj, theta_deg):
    theta = np.deg2rad(theta_deg)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    return traj @ rot_matrix.T

def RB_FastDTW(query_traj, db_traj, rotation_steps=12):
    query_traj_normalized = normalize_trajectory(query_traj)
    db_traj_normalized = normalize_trajectory(db_traj)

    min_distance = float('inf')
    for angle in np.linspace(-30, 30, rotation_steps, endpoint=False):
        rotated_traj = rotate_trajectory(query_traj_normalized, angle)
        distance, _ = fastdtw(rotated_traj, db_traj_normalized)
        if distance < min_distance:
            min_distance = distance
    reverse_query_traj = query_traj_normalized[::-1]
    for angle in np.linspace(-30, 30, rotation_steps, endpoint=False):
        rotated_traj = rotate_trajectory(reverse_query_traj, angle)
        distance, _ = fastdtw(rotated_traj, db_traj_normalized)
        if distance < min_distance:
            min_distance = distance
    
    return min_distance

class TraceDataset:
    def __init__(self, dataset_infos_path):
        self.dataset_infos = []
        for dataset_path in dataset_infos_path:
            dataset_infos = np.load(dataset_path, allow_pickle=True)['traces']
            self.dataset_infos.extend(dataset_infos)
        
        self.traces = []
        self.instructions = []
        for data_info in dataset_infos:
            self.traces.append(data_info['trace_trace'])
            self.instructions.append(data_info['task_instruction'])
            self.dataset_infos.append(data_info)

        print(len(self.traces)) 
        print(len(self.instructions))
        print(len(self.dataset_infos))
        
        self._precompute_clip_embeddings()
    
    def _precompute_clip_embeddings(self):
        clip_inputs = clip_tokenizer(self.instructions, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**clip_inputs)
            self.clip_embeds = outputs["pooler_output"].cpu().numpy()
        print(self.clip_embeds.shape)

    def __len__(self):
        return len(self.instructions)

class CrossModalRetriever:
    def __init__(self, dataset, alpha=0.0):
        self.dataset = dataset
        self.alpha = alpha
    
    def retrieve_top_k(self, query_text, query_trace, k=5):
        clip_inputs = clip_tokenizer(
            [query_text], 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = clip_model(**clip_inputs)
            query_embed = outputs["pooler_output"].cpu().numpy()[0]
        
        clip_distances = []
        dtw_distances = []
        for idx in range(len(self.dataset)):
            clip_dist = cosine(query_embed, self.dataset.clip_embeds[idx])
            clip_distances.append(clip_dist)
        
            db_trace = self.dataset.traces[idx]
            dtw_dist = RB_FastDTW(query_trace, uniform_sample(db_trace))
            dtw_distances.append(dtw_dist)
        
        clip_max = max(clip_distances) if clip_distances else 1e-6
        dtw_max = max(dtw_distances) if dtw_distances else 1e-6
        
        scores = []
        for idx in range(len(self.dataset)):
            clip_norm = clip_distances[idx] / clip_max
            dtw_norm = dtw_distances[idx] / dtw_max
            combined = self.alpha * clip_norm + (1 - self.alpha) * dtw_norm
            scores.append( (combined, idx) )
        
        top_k = nsmallest(k, scores, key=lambda x: x[0])
        return [(
            score, idx,
            self.dataset.instructions[idx],
            self.dataset.traces[idx],
            self.dataset.dataset_infos[idx]
        ) for (score, idx) in top_k]

if __name__ == "__main__":
    dataset_path = ["Your dataset keypoint trace info path"]
    dataset = TraceDataset(dataset_path)
    retriever = CrossModalRetriever(dataset, alpha=0.0)
    
    # Simulated query
    query_trace = [(42, 154), (42, 126), (42, 98), (70, 98), (98, 98), (126, 98), (154, 98), (154, 126)]
    query_text = "place_wine_at_rack_location"
    
    results = retriever.retrieve_top_k(query_text, query_trace, k=3)
    
    print("Top-K Retrieved Items:")
    for score, idx, instr, trace, data_info in results:
        print(f"Score: {score:.4f} | Index: {idx}")
        print(f"Instruction: {instr}")
        print(f"Trace Shape: {trace.shape}\n")
        print(f"data_info: {data_info['trace_idx']}\n")