### UTMIST AI2 Technical Guide

This guide shows you how to navigate the repository, implement your agent, and submit to our GitHub Actions pipelines for validation and battles. 

### Quick Repo Tour

- **environment/**: Game engine and helpers used to run matches
  - `environment/agent.py`: Base `Agent` API and helpers (e.g., `run_match`, `CameraResolution`, `gen_reward_manager`)
  - `environment/environment.py`, `pvp_match.py`, `constants.py`: Core environment logic and match runner
  - `environment/assets/…`: Sprites, sounds, and effects
- **user/**: Your workspace for submissions
  - `user/my_agent.py`: Where you define your `SubmittedAgent(Agent)`
  - `user/validate.py`: Local/CI validation entrypoint (used by Validation pipeline)
  - `user/battle.py`: Loads two agents and runs a match (used by Battle pipeline)
- **.github/workflows/**: GitHub Actions pipelines
  - `agent_validation_pipeline.yaml`: Validates a single submission
  - `agent_battle_pipeline.yaml`: Battles two submissions
- **install.sh**, `requirements.txt`: Setup and dependencies used locally and in CI
- **guides/**: This guide and other documentation
- **rl-model/** and `rl-model.zip`: Sample model artifacts

### Your Agent: Minimum Requirements

- Implement `SubmittedAgent(Agent)` in `user/my_agent.py`.
- The class must be importable without extra interaction and must follow the `Agent` interface.
- Please make sure your weights are publicly downloadable (e.g., `gdown` with a public link).

Minimal skeleton you can adapt:

```python
from environment.agent import Agent

class SubmittedAgent(Agent):
    def __init__(self, file_path=None):
        super().__init__(file_path)

    def _initialize(self) -> None:
        # Initialize your policy/model here
        # If you won't train further inside the Agent, you can free the env:
        # del self.env
        pass

    def predict(self, obs):
        # Return an action for the given observation
        return 0

    def save(self, file_path: str) -> None:
        pass
```

NOTE: If you use pretrained weights, implement `_gdown()` to fetch them, then load in `_initialize()`.

### BONUS: Implementing your agent in `ttnn` (Tenstorrent neural network library)

This year, we are very honoured to partner with Tenstorrent to bring this tournament to life. Their RISC-V based
accelerator hardware makes AI inference workloads memory efficient and low-latency. We have set up a Tenstorrent competition track 
for those interested in exploring new hardware options. `ttnn` is an operator library that calls specialized kernels written to do deep learning computations 
efficiently on our hardware. Without going too deep, you can think of `ttnn` as Tenstorrent's PyTorch. You can perform matrix multiplications, elementwise additions, and much more
ops using `ttnn`. As of today, `ttnn` only supports inferencing workloads, training is not supported in `ttnn` but is a part of the `tt-train` software stack. `tt-train` is still very experimental and may introduce a lot more roadblocks for this tournament so it will be out of our scope of support for this tournament.

Because of this limitation, we recommend participants to train your RL agents using PyTorch (locally or on colab or on your own host CPU) then bringing up a functional version with trained torch weights using `ttnn`. We encourage participants to get involved with this track since it is a good learning experience for learning about state of the art AI accelerators. For detailed step by step guide to bringing up your agent using `ttnn` please refer to `tt_guide.md`.


### Getting Started Locally 

0) Create a fork of the UTMIST-AI2 repository and clone the repostiory locally
```
git clone https://github.com/<your username>/UTMIST-AI2
```

1) Install deps

```bash
pip install -r requirements.txt
```

2) Develop your agent in `my_agent.py`

3) Validate locally (single-agent smoke test). Produces `validate.mp4` in repo root by default.

```bash
pytest -s user/validate.py
```

4) Battle locally (two agents on your machine). Point both env vars to agent files.

```bash
AGENT1_PATH=user/my_agent.py AGENT2_PATH=user/my_agent.py pytest -s user/battle.py
```

### Validation Pipeline (CI)
In order to accept and qualify your submission, one must first validate their submission by running the validation pipeline in Github Actions. Validation serves as a functionality check for your agent, and is completed done by running your agent against a dummy agent to check that your agent can in fact successful complete a match. A submission is not accepted unless there is a successful validation pipeline recorded. We do not recommend doing debugging through Github Actions as it wastes compute and memory resources. Please test your agent thoroughly locally before launching a validation pipeline run. If our team detects unreasonable and suspicious amount of pipeline runs, we're going to have a serious talk with your team.

How to run it:
- Go to the UTMIST-AI2 repository’s Actions tab → "RL Tournament Validation Pipeline" → "Run workflow"
- Choose main as the branch to run and enter your GitHub username (fork owner username) and start the run
- Your submission will be counted as validated if it successfully completes the dummy match in the run.

How to see results:
- Open the run → check the live logs for messages like:
  - "Warming up your agent …"
  - "Validation match has started …"
  - "Validation match has completed successfully!"
- If your code writes files under `submission/results/`, they will be available as a downloadable artifact named `results-<username>`.

Notes:
- The default `user/validate.py` writes the video to `validate.mp4` in the working directory.
- To persist outputs in CI, write copies into `results/`, e.g., `submission/results/validate.mp4`.

### Battle Pipeline (CI)
Each team has the capability to launch battles with other participants in the tournamnet. Our team enables this capability also through Github Actions. 

- File: `.github/workflows/agent_battle_pipeline.yaml`
- Trigger: Manual `workflow_dispatch` with inputs `username1` and `username2` (must be different)
- What it does:
  - Checks out the main repo (branch `aling/battle`)
  - Runs `./install.sh`
  - Clones two forks: `https://github.com/<username1>/UTMIST-AI2` and `https://github.com/<username2>/UTMIST-AI2`
  - Copies each fork’s `user/my_agent.py` into `agents/<usernameX>/my_agent.py`
  - Runs `pytest -s user/battle.py` with env vars:
    - `AGENT1_PATH=agents/<username1>/my_agent.py`
    - `AGENT2_PATH=agents/<username2>/my_agent.py`
- Timeouts: The battle test has a 300-second timeout.

How to run it:
- Actions tab → "RL Tournament Battle Pipeline" → "Run workflow"
- Provide two different GitHub usernames and start the run
- 

How to see results:
- Open the run → watch logs for lines like:
  - "✅ Both agents successfully instantiated."
  - `Agent1 vs Agent2`
  - "Battle has completed successfully!"
- The default test writes `battle.mp4` to the workspace but the workflow does not upload artifacts. If you need saved videos in CI, ask maintainers to enable artifact upload for the battle job, or run battles locally.


### Common Issues & Fixes

- **ImportError: SubmittedAgent not found**: Ensure the class name is exactly `SubmittedAgent` in `user/my_agent.py`.
- **Timeouts**: Trim model init; no training is supported in our CI pipeline; ensure your model is lightweight.
- **Dependency errors**: Add required packages to `requirements.txt` and use APIs available in the environment image (Python 3.10).
- **Path issues in Battle**: The pipeline sets `AGENT1_PATH`/`AGENT2_PATH` for you; locally, export them before running `user/battle.py`.

### Final reminders
- Make sure that `user/my_agent.py` defines `SubmittedAgent(Agent)` and runs locally
- Ensure that `pytest -s user/validate.py` works and finishes < 60s
- Remeber to test everything locally first and trigger the appropriate workflow in Actions and monitor logs
- Our internal team reserves the right to terminate jobs if we observe suspicious launches or behaviour. 
- Be respectful towards others within the UTMIST community 

If you have any questions please do not hesitate to ask the internal team of AI2. Happy hacking, and good luck in the tournament!


