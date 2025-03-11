'''
key_trace_finder.py
Given a series of images, return the trace of keypoint

pipeline:
    1. trace detection
    (part of the code following Paper co-tracker(https://github.com/facebookresearch/co-tracke), 
                            and TraceVLA(https://github.com/umd-huang-lab/tracevla))
    2. return the longest trace as the keypoint trace
    (we use shi-tomasi corner detection to reduce the usage of GPU memory)
'''
import numpy as np
import torch
from PIL import Image, ImageDraw
import os
import cv2
from typing import Tuple
from cotracker.predictor import CoTrackerPredictor
if torch.cuda.is_available():
    torch.cuda.empty_cache()

class KeyTraceFinder:
    def __init__(self, cotracker_model_path: str, device: str = 'cuda:0', max_points: int = 40, candidate_points: int = 20):
        self.device = device
        self.max_points = max_points
        self.candidate_points = candidate_points
        self.model = CoTrackerPredictor(
            checkpoint=os.path.join(cotracker_model_path)
        ).to(device)
        if torch.cuda.is_available():
           torch.cuda.empty_cache()

    def _preprocess_with_keypoints(self, image_list: list[Image.Image]) -> tuple[torch.Tensor, np.ndarray]:
        # Calculate the area that has been moved
        
        prev_frame = np.array(image_list[0].convert('L')) 
        last_frame = np.array(image_list[int(len(image_list)-1)].convert('L'))
        diff_mask = self._get_motion_mask(prev_frame, last_frame)
        
        processed = []
        for img in image_list:
            img_array = np.array(img).transpose(2, 0, 1)
            processed.append(img_array)

        mask_tensor = torch.from_numpy(diff_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
        video_tensor = torch.from_numpy(np.stack(processed)).float().to(self.device).unsqueeze(0)
        return video_tensor, mask_tensor

    def _get_motion_mask(self, prev_gray: np.ndarray, last_gray: np.ndarray) -> np.ndarray:
        """Generate binary masks for motion regions"""
        
        diff = cv2.absdiff(prev_gray, last_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Morphological processing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return cleaned

    def find_key_trace(self, image_list: list[Image.Image]) -> np.ndarray:
        if torch.cuda.is_available():
           torch.cuda.empty_cache()
        if len(image_list) < 2:
            return np.zeros((0, 2))
        img_size = image_list[0].size
        
        video_tensor, mask_tensor = self._preprocess_with_keypoints(image_list)
        with torch.no_grad():
            pred_tracks, _ = self.model(video_tensor, grid_size=30, segm_mask=mask_tensor)
            tracks = pred_tracks[0].cpu().numpy()  # (T, N, 2)

        return self._select_longest_trace(tracks, img_size)
    

    def _select_longest_trace(self, tracks: np.ndarray, img_size) -> np.ndarray:
        if tracks.shape[1] == 0:
            return np.zeros((0, 2))
        
        # hyper-parameters
        JUMP_THRESH = 5.0
        MIN_TRACE_POINTS = 5

        valid_traces = self._filter_valid_traces(
            tracks, 
            jump_thresh=JUMP_THRESH,
            min_points=MIN_TRACE_POINTS
        )
        if len(valid_traces) < 1:
            return np.zeros((0, 2))
        
        # select candidate trajectories
        candidates = np.array(self._select_top_candidates(valid_traces))
        best_idx = self._find_optimal_trace(tracks, candidates, img_size)
        
        return tracks[:, best_idx]

    def _filter_valid_traces(self, tracks, jump_thresh, min_points):
        """Filter out effective trajectories without mutations or oscillations"""
        valid_traces = []
        
        for idx in range(tracks.shape[1]):
            trace = tracks[:, idx]  # (T, 2)
            if len(trace) < min_points:
                continue
            
            displacements = self._calculate_displacements(trace)
            
            is_valid = True
            is_valid &= not self._detect_sudden_jump(displacements, jump_thresh)

            if is_valid:
                sampled_trace = self._uniform_sample(trace)
                trace_length = np.sum(self._calculate_displacements(sampled_trace))
                valid_traces.append( (trace_length, idx) )
        
        return np.array(valid_traces)
    
    def _calculate_displacements(self, trace):
        """Calculate the displacement between adjacent points"""
        return np.sqrt(np.sum(np.diff(trace, axis=0)**2, axis=1))

    def _uniform_sample(self, trace, num_points=10):
        """Uniform sampling trajectory points"""
        indices = np.linspace(0, len(trace)-1, num_points, dtype=int)
        return trace[indices]

    def _detect_sudden_jump(self, displacements, threshold):
        """Detecting sudden displacement"""
        if len(displacements) == 0:
            return False
        avg = np.mean(displacements)
        max_jump = np.max(displacements) / (avg + 1e-6)
        return max_jump > threshold

    def _select_top_candidates(self, valid_traces, max_candidates=10):
        """Select the candidate trajectory with the longest length"""
        sorted_traces = sorted(valid_traces, key=lambda x: -x[0])
        K = min(max_candidates, len(sorted_traces))
        return [x[1] for x in sorted_traces[:K]]

    def _find_optimal_trace(self, tracks, candidates, img_size):
        """Select the trace with the most frames left in the image from the candidate set"""
        best_idx = candidates[0]
        max_valid = -1
        
        for idx in candidates:
            idx = int(idx)
            trace = tracks[:, idx]
            valid_count = self._count_valid_frames(trace, img_size)
            
            if valid_count > max_valid:
                max_valid = valid_count
                best_idx = idx
                
        return best_idx

    def _count_valid_frames(self, trace, img_size):
        """Count the effective frame within the image area"""
        h, w = img_size
        x_valid = (trace[:, 0] >= 0) & (trace[:, 0] < w)
        y_valid = (trace[:, 1] >= 0) & (trace[:, 1] < h)
        return np.sum(x_valid & y_valid)
    
    def visualize_trace(self,
                    image_list: list[Image.Image],
                    trace: np.ndarray,
                    output_path: str = "trace_visualization.gif",
                    show_points: bool = True,
                    line_width: int = 2,
                    margin_ratio: float = 0.5) -> None:
        """
        Expand the canvas to display the complete trajectory
        """
        base_width, base_height = image_list[0].size
        margin = int(base_height * margin_ratio)
        canvas_width = base_width + 2 * margin
        canvas_height = base_height + 2 * margin
        
        vis_frames = []
        for i in range(len(image_list)):
            frame = self._draw_direct_frame(
                image_list[i], 
                trace[:i+1],
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                base_width=base_width,
                base_height=base_height,
                margin=margin,
                show_points=show_points,
                line_width=line_width
            )
            vis_frames.append(frame)

        vis_frames[0].save(
            output_path,
            save_all=True,
            append_images=vis_frames[1:],
            duration=100,
            loop=0
        )

    def _draw_direct_frame(self,
                        orig_img: Image.Image,
                        current_trace: np.ndarray,
                        canvas_width: int,
                        canvas_height: int,
                        base_width: int,
                        base_height: int,
                        margin: int,
                        show_points: bool,
                        line_width: int) -> Image.Image:
        canvas = Image.new("RGB", (canvas_width, canvas_height), (60, 60, 60))
        draw = ImageDraw.Draw(canvas)
        
        canvas.paste(orig_img, (margin, margin))
        draw.rectangle([margin, margin, margin+base_width, margin+base_height], outline=(255,255,255), width=2)
        
        # Convert coordinates to canvas coordinate system (located in the upper left corner of the original image at (margin, margin))
        canvas_trace = current_trace + [margin, margin]
        
        for i in range(1, len(canvas_trace)):
            start = tuple(canvas_trace[i-1])
            end = tuple(canvas_trace[i])
            
            if self._cross_visible_area(start, end, margin, base_width, base_height):
                draw.line([start, end], fill=(0, 255, 0), width=line_width)
            else:
                draw.line([start, end], fill=(255,165,0), width=line_width)
        
        if show_points and len(canvas_trace) > 0:
            latest = tuple(canvas_trace[-1])
            draw.ellipse([latest[0]-4, latest[1]-4, latest[0]+4, latest[1]+4], fill=(255,0,0))
        
        return canvas

    def _cross_visible_area(self, 
                        point1: tuple[float], 
                        point2: tuple[float],
                        margin: int, 
                        base_width: int,
                        base_height: int) -> bool:
        """Determine whether the line segment passes through the visible area"""
        visible_rect = (margin, margin, margin+base_width, margin+base_height)
        
        if (max(point1[0], point2[0]) < visible_rect[0] or
            min(point1[0], point2[0]) > visible_rect[2] or
            max(point1[1], point2[1]) < visible_rect[1] or
            min(point1[1], point2[1]) > visible_rect[3]):
            return False
        return True
    