from environment.environment import RenderMode, CameraResolution
from environment.agent import run_real_time_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
from user.opp_agent import SubmittedAgent as opp
import pygame
pygame.init()

my_agent = UserInputAgent()

#Input your file path here in SubmittedAgent if you are loading a model:
opponent = SubmittedAgent(file_path=None)

match_time = 999999

# Run a single real-time match
run_real_time_match(
    agent_1=opponent,  # Your AI
    agent_2=opp(file_path="checkpoints/v2.zip"),  # You
    #agent_2=my_agent,
    max_timesteps=30 * 9999,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)
