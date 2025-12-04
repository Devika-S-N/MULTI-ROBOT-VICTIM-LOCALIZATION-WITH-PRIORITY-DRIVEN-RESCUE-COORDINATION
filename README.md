# MULTI ROBOT VICTIM LOCALIZATION WITH PRIORITY DRIVEN RESCUE COORDINATION

A simulation framework for coordinating multi-robot teams in disaster response scenarios. The system uses decentralized consensus mechanisms to identify victim locations while filtering out false positives, followed by collaborative navigation to reach identified victims. This work helps in analysing the Global consensus using Distributed Consensus and Local consensus based on Rendezvous Teams.

## Project Overview

This project simulates a swarm of robots conducting search and rescue operations in a continuous 2D environment. The simulation is divided into two main phases:

1. **Phase 1: Global Distributed Consensus** - Robots perform random walks while sharing sensor information to reach consensus on victim locations through a noisy, distributed consensus algorithm.

2. **Phase 2: Local Rendezvous Consensus** - Robots navigate toward identified victims using swarm-based attraction and repulsion forces while maintaining group cohesion and performs consensus on victim location within the teams.

## Project Structure

```
├── main.py                    # Entry point for the simulation
├── environment.py             # World configuration and site placement
├── robot.py                   # Robot state and dynamics
├── phase1_markov.py          # Phase 1: Global Distributed Consensus
├── phase2_navigation.py       # Phase 2: Local Rendezvous Consensus
├── animation.py              # Video generation from simulation logs
├── utils.py                  # Analysis and plotting utilities
├── phase2_log.csv            # Sample Phase 2 simulation log
└── simulation_plots/         # Generated visualization outputs
    ├── combined/             # Combined phase analysis plots
    ├── phase_1_video/        # Phase 1 animation videos
    ├── phase_2_video/        # Phase 2 animation videos
    ├── phase1/               # Phase 1 analysis data
    └── phase2/               # Phase 2 analysis data
```

## Key Features

### Phase 1: Global Distributed Consensus
- **Random Walk Search**: Robots perform noisy random walks to explore the environment
- **Distributed Consensus**: Robots share sensor measurements and reach consensus on victim positions through iterative averaging with gains based on confidence
- **Sensor Noise**: Models realistic measurement noise and false positive confusion
- **Position Tolerance**: Consensus is achieved when robots' position estimates reach specified convergence criteria

### Phase 2: Local Rendezvous Consensus
- **Swarm Cohesion**: Neighbor-based attraction and repulsion forces keep robots grouped
- **Victim Attraction**: Robots are attracted to identified victim locations
- **False Positive Repulsion**: Weaker attraction to false positives prevents overcommitment
- **Rendezvous Consensus**: Consensus is achieved when robots' position estimates reach specified convergence criteria within the rendezvous team.
- **Collision Avoidance**: Local repulsion between robots and site boundaries
- **Wall Avoidance**: Robots are repelled from world boundaries

### Configuration
- Customizable world size, robot count, and victim/false positive distribution
- Adjustable sensor ranges, communication ranges, and movement speeds
- Tunable consensus parameters and force gains for different scenarios
- Optional fixed random seed for reproducibility

## Installation

### Requirements
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- OpenCV (for video generation)
- Openpyxl (for Excel export)

### Setup

```bash
# Install dependencies
pip install numpy pandas matplotlib opencv-python openpyxl

# Navigate to project directory
cd /path/to/project
```

## Usage

### Basic Simulation

```bash
# Run with default parameters (20 robots, 2 victims, 10 false positives)
python main.py

# Run with custom parameters
python main.py --robots 50 --victims 3 --false 15 --seed 42

# Specify output filenames
python main.py --phase1_video my_phase1.mp4 --phase2_video my_phase2.mp4 \
               --phase1_excel phase1_results.xlsx --phase2_excel phase2_results.xlsx
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--robots` | int | 20 | Number of robots in the swarm |
| `--victims` | int | 2 | Number of true victim locations |
| `--false` | int | 10 | Number of false positive locations |
| `--seed` | int | None | Random seed for reproducibility |
| `--phase1_video` | str | phase1_random_walk.mp4 | Phase 1 output video filename |
| `--phase2_video` | str | phase2_navigation.mp4 | Phase 2 output video filename |
| `--combined_video` | str | combined_phase1_phase2.mp4 | Combined output video filename |
| `--phase1_excel` | str | phase1_results.xlsx | Phase 1 results Excel file |
| `--phase2_excel` | str | phase2_results.xlsx | Phase 2 results Excel file |
| `--phase2_mode` | str | continue | "continue" (start from Phase 1 end) or "reset" (return to initial positions) |


## Output Files

The simulation generates the following outputs:

- **Excel Files**: Summary statistics and per-robot metrics for each phase
- **Videos**: MP4 animations of robot trajectories, detection progress, and navigation
- **CSV Logs**: Detailed timestep-by-timestep logs for analysis
- **Analysis Data**: NumPy arrays with detection delays, errors, and belief state metrics

## Configuration Examples

### Large Swarm Search
```bash
python main.py --robots 100 --victims 5 --false 30 --seed 123
```

### Small Disaster Zone
```bash
python main.py --robots 10 --victims 1 --false 3 --seed 456
```