# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM

# To run the sample TTNN model, you can uncomment the 2 lines below: 
# import ttnn
# from user.my_agent_tt import TTMLPPolicy


class SubmittedAgent(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0
        self.prev_pos = None
        self.down = False
        self.recover = False

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()
        facing = self.obs_helper.get_section(obs, 'player_facing')

        opp_grounded = self.obs_helper.get_section(obs, 'opponent_grounded')
        opp_state = self.obs_helper.get_section(obs, 'opponent_state')
        opp_move_type = self.obs_helper.get_section(obs, 'opponent_move_type')

        is_opponent_spamming = opp_grounded == 1 and opp_state == 8 and opp_move_type > 0

        spawners = self.env.get_spawner_info()

        # pick up a weapon if near
        '''
        if self.obs_helper.get_section(obs, 'player_weapon_type') == 0:
            for w in spawners:
                if euclid(pos, w[1]) < 3:
                    action = self.act_helper.press_keys(['h'], action)
        '''
                    
        # emote for fun
        if self.time == 10 or self.obs_helper.get_section(obs, 'opponent_stocks') == 0:
            action = self.act_helper.press_keys(['g'], action)
            return action

        if self.prev_pos is not None:
            self.down = (pos[1] - self.prev_pos[1]) > 0
        self.prev_pos = pos

        self.recover = False
        if pos[0] < -4.8:
            action = self.act_helper.press_keys(['d'], action)
            self.recover = True
        elif pos[0] > -4.2 and pos[0] < 0:
            action = self.act_helper.press_keys(['a'], action)
            self.recover = True
        elif pos[0] > 0 and pos[0] < 4.2:
            action = self.act_helper.press_keys(['d'], action)
            self.recover = True
        elif pos[0] > 4.8:
            action = self.act_helper.press_keys(['a'], action)
            self.recover = True

        # Jump if falling
        if pos[1] > -5 and (self.down or (self.obs_helper.get_section(obs, 'player_grounded') == 1) and not is_opponent_spamming):
            if self.time % 10 == 0:
                action = self.act_helper.press_keys(['space'], action)
            if self.recover and self.obs_helper.get_section(obs, 'player_grounded') == 0 and self.obs_helper.get_section(obs, 'player_jumps_left') == 0 and self.obs_helper.get_section(obs, 'player_recoveries_left') == 1 and self.time % 2 == 0:
                action = self.act_helper.press_keys(['k'], action)

        
        if not self.recover:
            if opp_pos[0] > pos[0]:
                action = self.act_helper.press_keys(['d'], action)
            elif opp_pos[0] < pos[0]:
                action = self.act_helper.press_keys(['a'], action)
        
                
        # Attack if near
        if not self.recover and abs(pos[0] - opp_pos[0]) < 0.5 and pos[1] < opp_pos[1]:
            action = self.act_helper.press_keys(['s'], action)
            action = self.act_helper.press_keys(['k'], action)
        elif not self.recover and euclid(pos, opp_pos) < 4:
            action = self.act_helper.press_keys(['j'], action)

        return action

def euclid (a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2
