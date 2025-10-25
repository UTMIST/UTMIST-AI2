import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.environment import RenderMode, CameraResolution
from environment.agent import run_real_time_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame
pygame.init()

my_agent = UserInputAgent()
my_agent2 = UserInputAgent()

#Input your file path here in SubmittedAgent if you are loading a model:
opponent = SubmittedAgent()

match_time = 99999

# Run a single real-time match
run_real_time_match(
    agent_1=my_agent,
    agent_2=my_agent2,
    max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)