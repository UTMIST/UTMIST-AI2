### UTMIST AI2 TTNN Guide
Author: Ambrose Ling

This guide will show how to bring up your own agent in ttnn from scratch. TNN is a operator library that allows you to leverage tenstorrent hardware. We will demonstrate a way that allows you to bring up your own neural network architecture in Torch with training integration with SB3 in torch, following that you can then bring up a ttnn version of the model, and use that during inference.

1) Define your custom neural network that acts as your feature extractor in torch. You can find the following code snippet in `user/train_agent.py`. For this example, we construct a simple Multi Layer Perceptron (3 linear layers / linear transformations). This feature extractor takes in an input vector representing your observation space, then outputs a vector representing your action space. Your custom neural network is instiated inside `MLPExtractor`. This class acts as the interface to SB3, where you must inherit certain functions like `get_policy_kwargs` and `forward`. In the forward function, that is where you actually call forward pass on your custom neural network by doing `self.model(obs)`. 


```python
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP Base Features Extractor
    '''
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0], 
            action_dim=10,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim) #NOTE: features_dim = 10 to match action space output
        )
    
class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)
    
    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, policy_kwargs=self.extractor.get_policy_kwargs(), verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )
```

2) Once you have the above set up, you are ready to train your model. You can run training by instantiating the CustomAgent and passing it the BaseFeaturesExtractor which uses your custom neural network. This is completely decoupled with how you design your reward function. So you are free to get creative with other parts of the training process on top of the neural network architecture. You can run training with the following script and it will continously save your agent's weights. In the following save setting handler it would save to `checkpoints/tt_mlp/` folder.

```python
if __name__ == '__main__':
    # Create agent
    my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)

    # Reward manager
    reward_manager = gen_reward_manager()
    # Self-play settings
    selfplay_handler = SelfPlayRandom(
        partial(type(my_agent)), # Agent class and its keyword arguments
                                 # type(my_agent) = Agent class
    )

    # Set save settings here:
    save_handler = SaveHandler(
        agent=my_agent, # Agent to save
        save_freq=100_000, # Save frequency
        max_saved=40, # Maximum number of saved models
        save_path='checkpoints', # Save path
        run_name='tt_mlp',
        mode=SaveHandlerMode.FORCE # Save mode, FORCE or RESUME
    )

    # Set opponent settings here:
    opponent_specification = {
                    'self_play': (8, selfplay_handler),
                    'constant_agent': (0.5, partial(ConstantAgent)),
                    'based_agent': (1.5, partial(BasedAgent)),
                }
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    train(my_agent,
        reward_manager,
        save_handler,
        opponent_cfg,
        CameraResolution.LOW,
        train_timesteps=100_000,
        train_logging=TrainLogging.PLOT
    )
```

3) Once you have a trained agent and you have weights saved locally, you can start bringing up the `ttnn` version of your custom neural network. The goal of this step is to implement a 1-to-1 model that uses ttnn APIs that run the same computation. `ttnn` supports a lot of the same APIs that exist in torch. For which operators are supported and what ones to use you can check the `ttnn` documentation [here](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html). To write the same MLP model in ttnn, we have to retrieve the weights of our torch model by passing in the torch state_dict. These tensors (weights and biases) will be converted to `ttnn` tensors. We also define our forward pass that computes matrix multiplications of your input with the weights. A note is that your TTNN model should inherit the `torch.nn.Module` class and must implement a `forward` function, so you must convert input tensors to ttnn tensors and the output from ttnn tensors back to torch tensors since SB3 sees this module as a black box with torch inputs and torch outputs.

```python
import ttnn
class TTMLPPolicy(nn.Module):
    def __init__(self, state_dict, mesh_device):
        super(TTMLPPolicy, self).__init__()
        # Define a pointer to your device (similar to torch device)
        self.mesh_device = mesh_device
        # This extracts the weights matrix from each linear layer using the torch state dict and
        # converts it from a torch tensor to a ttnn tensor, we use the 
        # default DRAM memory configuration and uses TILE LAYOUT 
        # meaning that in memory the tensor is organized as tiles (think of it as 32x32 datums) 
        self.fc1 = ttnn.from_torch(
            state_dict["fc1.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.fc2 = ttnn.from_torch(
            state_dict["fc2.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.fc3 = ttnn.from_torch(
            state_dict["fc3.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        self.fc1_b = ttnn.from_torch(state_dict["fc1.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.fc2_b = ttnn.from_torch(state_dict["fc2.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.fc3_b = ttnn.from_torch(state_dict["fc3.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        # We convert the tensor to bfloat16 data type
        obs = obs.to(torch.bfloat16)
        # Convert the input to a ttnn tensor to a torch tensor
        tt_obs = ttnn.from_torch(obs, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)

        # Performs a linear layer similar to torch.nn.Linear so the input is matmuled with the weights
        # x1 = relu(tt_obs @ self.fc1 + fc1_b) <- equivalent formula
        x1 = ttnn.linear(tt_obs, self.fc1, bias=self.fc1_b, activation="relu")
        tt_obs.deallocate()

        x2 = ttnn.linear(x1, self.fc2, bias=self.fc2_b, activation="relu")
        x1.deallocate()

        x3 = ttnn.linear(x2, self.fc3, bias=self.fc3_b)
        x2.deallocate()

        # Convert the result from ttnn tensor to a torch tensor and back to torch float 32 data type
        tt_out = ttnn.to_torch(x3).flatten().to(torch.float32)

        return tt_out
```

4) In order to validate that your TTNN model is implemented correctly we recommend doing what is performed inside `user/my_agent_tt.py`. You have both your torch model and ttnn model implemented. You pass the same input to both models the output from your ttnn model and the output from your torch model should be very very close. Please check that out to see how we set up inference on both models and compare the outputs. You can do this by performing a PCC check (we compute the Pearson Correlation Coefficient between the torch model output and ttnn model output). The correct output should be above 0.99. We also put that code snippet below for how to do that.

```python
def test_mlp_policy():
    
    # Open the device (since we are only using single devices N150 cards, your mesh shape will be 1x1)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))
    
    # Dimensions based on our custom RL environment
    batch_size = 1
    action_dim = 10 
    hidden_dim = 64
    obs_dim = 64

    # Create torch input
    x = torch.randn(1, obs_dim, dtype=torch.bfloat16)
    # Create torch model
    policy = MLPPolicy(obs_dim, action_dim, hidden_dim)

    # Create TTNN model, we pass the torch model state dict to use its weights
    tt_policy = TTMLPPolicy(policy.state_dict(), mesh_device)

    # Run forward pass on torch model
    y = policy(x)

    # Run forward pass on ttnn model
    tt_y = tt_policy(x)

    # Check that the Pearson Correlation Coefficient is above 0.99 (meaning that these 2 tensors are very very close to eachother) to check for correctness
    if check_pcc(y, tt_y):
        print("✅ PCC check passed!")
    else:
        print("❌ PCC below threshold.")
```

5. Once you have validated that your ttnn model is correct, you can port it into your `SubmittedAgent` class. Here's an example of how to do that:
```python
import ttnn
class SubmittedAgent(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''
    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        super().__init__(file_path)
        # Defining your ttnn device pointer the same way we did in `my_agent_tt.py`
        self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = PPO("MlpPolicy", self.env, verbose=0)
            del self.env
        else:
            self.model = PPO.load(self.file_path)
        # HERE ->
        # At this point your self.model.policy points to your trained torch custom neural network 
        mlp_state_dict = self.model.policy.features_extractor.model.state_dict()
        # Here you define your tttnn model and we extract the state dictionary of your custom neural network and pass it to your ttnn model
        self.tt_model = TTMLPPolicy(mlp_state_dict, self.mesh_device)
        # Once you have a ttnn model, we make the following models point to your ttnn model
        # such that when you perform inference during a match, calling self.model(obs) will actually invoke the forward pass of our ttnn model
        self.model.policy.features_extractor.model = self.tt_model
        self.model.policy.vf_features_extractor.model = self.tt_model
        self.model.policy.pi_features_extractor.model = self.tt_model

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    # If modifying the number of models (or training in general), modify this
    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
```

5) Once your SubmittedAgent is set up, you can run a match to check that it works:
```python
from environment.environment import RenderMode, CameraResolution
from environment.agent import run_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame
pygame.init()

my_agent = UserInputAgent()

#Input your file path here in SubmittedAgent if you are loading a model:
# Use the path where you saved the weights of your trained model from the training above you started above
opponent = SubmittedAgent(file_path='checkpoints/tt_mlp/rl_model_769500_steps.zip')

match_time = 99999

# Run a single real-time match
run_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
    video_path='tt_agent.mp4'
)
```

6) When you submit your agent via the validation pipeline, you will have the option to click `Run Workflow`. There is a check box for if your SubmittedAgent uses ttnn, checking that box will make your job run on Github Action runners that have Tenstorrent hardware. Then click the green Run button. For battles pipeline, we will run them all on our designated runners that have Tenstorrent hardware so you can just type your username and your opponent's username and start the battle.

7) Thats it! The above steps should work for any custom neural network architecture. If you run into any issues with ttnn feel free to reach out to me. I look forward to helping all of you bringing your models to life in ttnn. Best of luck everyone!