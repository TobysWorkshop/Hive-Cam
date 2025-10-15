import os
import cv2
import torch
import numpy as np
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.transforms.functional import resize, to_tensor

def generate_raft_flow(video_path, output_folder, device="cuda"):
    # Load RAFT model (pretrained)
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=True).to(device)
    model = model.eval()

    os.makedirs(output_folder, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: cannot read video")
        return

    prev_tensor = to_tensor(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    frame_idx = 1
    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        next_tensor = to_tensor(cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

        # RAFT input prep
        h, w = prev_tensor.shape[-2:]
        new_h, new_w = (h // 8) * 8, (w // 8) * 8

        prev_resized = resize(prev_tensor, [new_h, new_w])
        next_resized = resize(next_tensor, [new_h, new_w])

        with torch.no_grad():
            # Forward flow (t -> t+1)
            list_of_flows_fwd = model(prev_resized, next_resized)
            flow_fwd = list_of_flows_fwd[-1][0].permute(1, 2, 0).cpu().numpy()

            # Backward flow (t+1 -> t)
            list_of_flows_bwd = model(next_resized, prev_resized)
            flow_bwd = list_of_flows_bwd[-1][0].permute(1, 2, 0).cpu().numpy()

        # Save both
        np.save(os.path.join(output_folder, f"flow_fwd_{frame_idx:06d}.npy"), flow_fwd)
        np.save(os.path.join(output_folder, f"flow_bwd_{frame_idx:06d}.npy"), flow_bwd)

        prev_tensor = next_tensor
        frame_idx += 1

    cap.release()
    print(f"Done! Saved {frame_idx-1} forward/backward flow maps in {output_folder}")


if __name__ == "__main__":
    video_path = "data/videos/videoB.mp4"
    output_folder = "data/raft_flows"
    generate_raft_flow(video_path, output_folder, device="cuda")
