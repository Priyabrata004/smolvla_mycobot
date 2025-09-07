import os, time, numpy as np, torch
from PIL import Image

# Model + dataset stats
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.utils import preprocess_observation

def load_image_or_dummy(path=None, size=(256, 256)):
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB").resize(size)
        return np.array(img, dtype=np.uint8)
    # fallback: random image for a smoke test
    return (np.random.rand(size[1], size[0], 3) * 255).astype(np.uint8)

def main():
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load dataset stats for normalization (commonly used with SmolVLA)
    ds_meta = LeRobotDatasetMetadata("lerobot/svla_so101_pickplace")  # public dataset
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", dataset_stats=ds_meta.stats)
    policy.to(device)
    policy.eval()

    # Build one observation (single image + robot state)
    img = load_image_or_dummy(path=None, size=(256, 256))  # replace with your own RGB frame if you want
    # Find the state dimension expected by the policy’s normalizer
    state_dim = policy.normalize_inputs.buffer_observation_state.mean.shape[-1]
    state = np.zeros((state_dim,), dtype=np.float32)       # if you don't have proprio yet

    # Preprocess to tensors the way LeRobot expects
    observation = {"pixels": img, "agent_pos": state}
    observation = preprocess_observation(observation)
    observation = {k: v.to(device, non_blocking=(device.type=="cuda")) for k, v in observation.items()}

    # A natural-language task prompt (zero-shot)
    observation["task"] = ["Pick up the red block and place it in the bowl"]

    # Inference
    with torch.inference_mode():
        t0 = time.time()
        action = policy.select_action(observation)  # returns a Tensor
        dt = time.time() - t0

    print("Action shape:", tuple(action.shape))
    print(f"Inference time: {dt*1e3:.1f} ms")
    print(action)

if __name__ == "__main__":
    main()
