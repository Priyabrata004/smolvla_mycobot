# smolvla_mycobot

Vision–Language–Action (VLA) integration for **myCobot 280** using **SmolVLA**, built with **ROS 2** and **Gazebo**.

This repository explores how a lightweight Vision–Language–Action model (SmolVLA) can be connected to a real robotic control stack to generate joint-level robot motion from camera observations and natural-language instructions. The project is research-oriented and focuses on understanding the limitations of zero-shot VLA deployment on non-benchmark robots.

---

## 📌 Project Motivation

Recent Vision–Language–Action models demonstrate impressive results on benchmark robotic platforms, but their behavior under **domain shift**, **embodiment mismatch**, and **control interface differences** is still not well understood.

This project investigates:
- How SmolVLA behaves when deployed zero-shot on myCobot 280
- How learned action representations translate to real robot joint control
- Why naive deployment leads to unstable or meaningless motion
- What is required to move toward fine-tuned, embodiment-aware VLA control

---

## ✨ Key Features

- Real-time SmolVLA inference using camera images and language prompts
- ROS 2 integration for perception, inference, and actuation
- Gazebo simulation support for rapid experimentation
- Heuristic action post-processing for safety and stability
- Research-focused evaluation of VLA failure modes

---

## 📁 Repository Structure

```text
smolvla_mycobot/
├── mycobot_bringup/          # Robot bringup and launch files
├── mycobot_description/      # URDF, meshes, and robot model
├── mycobot_gazebo/           # Gazebo simulation setup
├── mycobot_moveit_config/    # MoveIt configuration (optional planning)
├── mycobot_ros2/             # ROS 2 integration packages
├── smolvla_realtime.py       # Real-time SmolVLA inference + control node
├── utils/                    # Helper scripts and utilities
└── README.md

```

## 🧠 System Overview

The system implements a perception-to-action pipeline:

```text
Camera Image + Language Prompt
            ↓
     Preprocessing (Normalization)
            ↓
         SmolVLA
            ↓
   Action Post-processing
            ↓
     ROS 2 Joint Trajectory
            ↓
          Robot
```

## 🔍 How It Works

### 1. Perception
- Subscribes to RGB camera images from ROS 2
- Images are resized and normalized to match SmolVLA training statistics

### 2. Language Conditioning
- Natural-language instructions are passed as task prompts
- Text is tokenized internally by the SmolVLA policy

### 3. Model Inference
- SmolVLA predicts a short-horizon action vector
- Output actions are normalized and model-specific

### 4. Action Post-processing
- Action semantics inferred heuristically (absolute vs delta, radians vs degrees)
- Joint limits enforced to ensure safety
- Temporal smoothing applied to reduce jitter

### 5. Robot Control
- Commands are published as `JointTrajectory` messages
- Executed via ROS 2 controllers in simulation or on hardware

---

## ⚙️ Installation

### Prerequisites
- ROS 2 (Humble or later recommended)
- Gazebo Ignition Fortress
- Python 3.8+
- PyTorch
- LeRobot / SmolVLA dependencies

### Clone and Build

```bash
cd ~/ros2_ws/src
git clone https://github.com/Priyabrata004/smolvla_mycobot.git
cd ..
rosdep install -i --from-path src --rosdistro humble -y
colcon build --symlink-install
source install/setup.bash
```

## ▶️ Running the System

### Launch Gazebo Simulation

Navigate to mycobot_bringup/scripts
```bash
bash mycobot_280_gazebo.sh

```

### Launch Inference Script
Navigate to mycobot_system_tests/scripts
```bash
python smolvla_realtime.py
```

## ⚠️ Current Limitations

- Zero-shot SmolVLA is not aligned with the myCobot embodiment
- Action outputs require heuristic interpretation
- No fine-tuning on robot-specific data
- Open-loop control without corrective feedback

These limitations are intentional research observations and motivate future work.

---

## 🔬 Research Focus

This repository serves as an experimental platform to study:
- Embodiment mismatch in Vision–Language–Action models
- Action-space alignment between learned policies and robot controllers
- Domain gap between training datasets and real/simulated environments
- Practical challenges of deploying foundation models in robotics

---

## 🚀 Future Work

- Fine-tuning SmolVLA on myCobot-specific datasets
- End-effector or OSC-based action representations
- Multilingual instruction grounding (e.g., Hindi)
- Sim-to-real transfer experiments
- Closed-loop correction using visual feedback

---

## 📜 License

This project is released under an open-source license.  
Please refer to individual submodules for their respective licenses.

---

## 🙌 Acknowledgements

- SmolVLA and LeRobot community
- ROS 2 and Gazebo ecosystem
- Elephant Robotics myCobot platform
- Automatic Addison YT Channel
