# UR5 Inference Runbook

Goal: run the trained UR5 policy with the model on a GPU machine and the robot/cameras on the UR5 machine.

## Machine Roles

### GPU Machine

This machine loads the checkpoint and serves actions over WebSocket.

Responsibilities:
- Has a working NVIDIA GPU.
- Has the `openpi-private` repo.
- Has the trained checkpoint, for example:
  `checkpoints/pi05_ur5_lora/ur5_pick_place_v1/3000`
- Runs `scripts/serve_policy.py` on port `8000`.

It does not connect to the UR5 robot, cameras, or gripper.

### UR5 Machine

This machine is `ur5-mars2`. It talks to the physical hardware.

Responsibilities:
- Connects to UR5 with `ur_rtde`.
- Reads base and wrist cameras.
- Reads gripper state and sends gripper commands.
- Sends each observation to the GPU machine.
- Receives action chunks from the GPU machine.
- Executes low-speed UR5 actions.

It should not load the full model unless the local GPU/RAM setup is known to work.

## Step 1: Start Policy Server on the GPU Machine

On the GPU machine:

```bash
cd /path/to/openpi-private
source .venv/bin/activate

JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=src python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_ur5_lora \
  --policy.dir checkpoints/pi05_ur5_lora/ur5_pick_place_v1/3000
```

Keep this terminal open.

Find the GPU machine IP:

```bash
hostname -I
```

Example:

```text
192.168.1.50
```

## Step 2: Test Network from the UR5 Machine

On the UR5 machine:

```bash
python - <<'PY'
import socket

host = "192.168.1.50"  # TODO: replace with GPU machine IP
port = 8000

s = socket.create_connection((host, port), timeout=5)
print("connected")
s.close()
PY
```

If this fails, check:
- Both machines are on the same network or VPN.
- The GPU machine firewall allows port `8000`.
- `serve_policy.py` is still running.
- The IP address is correct.

## Step 3: Test Policy Client Without Robot Motion

On the UR5 machine, use the WebSocket client to send a fake observation.

```bash
cd /home/kenn/Desktop/openpi-private
source .venv/bin/activate

PYTHONPATH=src python - <<'PY'
import numpy as np
from openpi_client import websocket_client_policy

host = "192.168.1.50"  # TODO: replace with GPU machine IP
port = 8000

policy = websocket_client_policy.WebsocketClientPolicy(host, port)

obs = {
    "observation/state": np.zeros(7, dtype=np.float32),
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "prompt": "pick up the banana and place it in the blue plate",
}

out = policy.infer(obs)
print(out.keys())
print(out["actions"].shape)
print(out["actions"][0])
PY
```

Expected action shape:

```text
(10, 7)
```

This means:
- `10` = `action_horizon`
- `7` = 6 UR5 joints + gripper

## Step 4: Dry-Run with Real Sensors

Before moving the robot, write or run a UR5 client that:
- Connects to UR5.
- Reads current `qpos` plus gripper into a 7D state.
- Reads base and wrist RGB images.
- Sends the observation to the policy server.
- Prints the returned action.
- Does not execute the action.

Observation format:

```python
obs = {
    "observation/state": state_7d,
    "observation/image": base_rgb,
    "observation/wrist_image": wrist_rgb,
    "prompt": task_prompt,
}
```

Policy output:

```python
actions = policy.infer(obs)["actions"]  # shape: (10, 7)
```

## Step 5: First Real Robot Test

Use a very conservative rollout:

- Start near the same home pose used during data collection.
- Keep the table layout similar to training.
- Use the exact training prompts, for example:
  - `pick up the banana and place it in the blue plate`
  - `pick up the tomato and place it in the red plate`
- Execute only the first action from each returned action chunk.
- Use low speed and short motion duration.
- Keep emergency stop reachable.
- Stop immediately if the robot moves unexpectedly.

Recommended first execution logic:

```python
actions = policy.infer(obs)["actions"]
action = actions[0]

# TODO: convert the 7D action into UR5 joint target + gripper command.
# TODO: send a small, low-speed command to the robot.
```

Use the one-step runner first:

```bash
cd /home/kenn/Desktop/openpi-private
source .venv/bin/activate

PYTHONPATH=src python examples/ur5/run_ur5_policy_step.py \
  --host 100.112.30.96 \
  --prompt "pick up the banana and place it in the blue plate" \
  --gripper-position 0.0 \
  --steps 1 \
  --max-joint-delta 0.03 \
  --speed 0.08 \
  --accel 0.08
```

This script waits for Enter before executing each step. Type `q` then Enter to stop without moving.

Do not move to an automatic closed-loop rollout until the action execution design is reviewed.

## Current Checkpoint

Current local checkpoint path on the UR5 machine:

```text
checkpoints/pi05_ur5_lora/ur5_pick_place_v1/3000
```

This is a `3000` step checkpoint. It is useful for testing the inference pipeline, but it may not be the final best robot policy.

If later checkpoints exist, also test:

```text
10000
20000
30000
```

Compare robot behavior rather than only training loss.
