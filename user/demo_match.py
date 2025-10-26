from environment.environment import RenderMode, CameraResolution
from environment.agent import run_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame
pygame.init()

my_agent = UserInputAgent()

#NOTE: Input your file path here in SubmittedAgent if you are loading a model:
opponent = SubmittedAgent()

match_time = 99999

# Run a single real-time match
run_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * match_time,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
    video_path='tt_agent.mp4' #NOTE: you can change the save path of the video here
)