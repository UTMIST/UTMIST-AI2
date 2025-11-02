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
import numpy as np
import random
import math
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
        file_path: Optional[str] = None,
    ):
        super().__init__(file_path)

        # To run a TTNN model, you must maintain a pointer to the device and can be done by 
        # uncommmenting the line below to use the device pointer
        # self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def _initialize(self) -> None:
        
        self.time = 0
        self.lastJumped = 0
        self.phase = "aggressive"
        self.platforms = [[2.5, 6.5, 0.4], [-6.5, -2.5, 2.4]]

        self.state_mapping = {
            'WalkingState': 0,
            'StandingState': 1,
            'TurnaroundState': 2,
            'AirTurnaroundState': 3,
            'SprintingState': 4,
            'StunState': 5,
            'InAirState': 6,
            'DodgeState': 7,
            'AttackState': 8,
            'DashState': 9,
            'BackDashState': 10,
            'KOState': 11,
            'TauntState': 12,
        }

        self.damage_last_time = 0
        self.state_reverse = {v: k for k, v in self.state_mapping.items()}
        self.can_cross = False
        self.last_dodged = 0
        self.used_attack = False
        self.last_platform = None

        self.weapons_data = {
            0: [
                {"keys": ["j"], "cover": [0, 5], "type": "ground", "range": "close"},
                {"keys": ["k"], "cover": [0, 2, 3, 5], "type": "ground", "range": "close"},
                {"keys": ["d", "j"], "cover": [0, 3, 5], "type": "ground", "range": "lunge"}, 
                {"keys": ["s", "j"], "cover": [0, 5], "type": "ground", "range": "far"},
                {"keys": ["d", "k"], "cover": [0, 3, 5], "type": "ground", "range": "lunge"},
                {"keys": ["s", "k"], "cover": [0, 4, 5], "type": "ground", "range": "close"},

                {"keys": ["j"], "cover": [0,5], "type": "aerial", "range": "close"}, 
                {"keys": ["d", "j"], "cover": [0, 5], "type": "aerial", "range": "far"},
                {"keys": ["k"], "cover": [0, 2, 3, 5], "type": "aerial", "range": "close"}, 
                {"keys": ["s", "k"], "cover": [5], "type": "aerial", "range": "lunge"}, 
                {"keys": ["s", "j"], "cover": [8], "type": "aerial", "range": "lunge"}, 
            ],  # Fist

            # HAMMER
            2: [
                {"keys": ["j"], "cover": [5], "type": "ground", "range": "close"},          # short swing, quick poke
                {"keys": ["k"], "cover": [2, 3, 5], "type": "ground", "range": "close"},    # big lunge, heavy hit
                {"keys": ["d", "j"], "cover": [5], "type": "ground", "range": "lunge"},     # step-in swipe (right)
                # {"keys": ["d", "k"], "cover": [2, 3, 5], "type": "ground", "range": "far"},     # heavy lunge (right)
                {"keys": ["s", "j"], "cover": [5], "type": "ground", "range": "far"},     # low sweep
                {"keys": ["j"], "cover": [1, 2, 3], "type": "aerial", "range": "far"},          # quick mid-air swing
                {"keys": ["s", "j"], "cover": [7], "type": "aerial", "range": "far"}, 
                {"keys": ["d","j"], "cover": [5], "type": "aerial", "range": "far"}, 
                {"keys": ["k"], "cover": [2], "type": "aerial", "range": "close"},
                {"keys": ["s", "k"], "cover": [7], "type": "aerial", "range": "lunge"},

                                      # big aerial smash
            ],
            # SPEAR
            1: [
                {"keys": ["j"], "cover": [5], "type": "ground", "range": "lunge"},          # short swing, quick poke
                {"keys": ["k"], "cover": [1, 2, 3], "type": "ground", "range": "lunge"},    # big lunge, heavy hit
                {"keys": ["d", "j"], "cover": [5, 3], "type": "ground", "range": "close"},
                {"keys": ["d", "j"], "cover": [5, 3], "type": "ground", "range": "far"},      # step-in swipe (right)
                {"keys": ["s", "j"], "cover": [8], "type": "ground", "range": "lunge"},     # low sweep
                {"keys": ["s", "k"], "cover": [3, 5], "type": "ground", "range": "close", "super_safe": True},     # downward spike
                {"keys": ["j"], "cover": [1, 2, 3, 4, 5, 6, 7, 8], "type": "aerial", "range": "close"},          # quick mid-air swing
                {"keys": ["d", "j"], "cover": [5], "type": "aerial", "range": "far"},          # quick mid-air swing
                {"keys": ["s", "j"], "cover": [7], "type": "aerial", "range": "far"},          # quick mid-air swing
                {"keys": ["k"], "cover": [1, 2, 3], "type": "aerial", "range": "lunge"},           # big aerial smash
                {"keys": ["s" ,"k"], "cover": [6, 7, 8], "type": "aerial", "range": "lunge"},
            ]
        }

        self.weapons_range = {
            0: 0,
            1: 0, # 0.6
            2: 0 # 0.3
        }

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        player_vel = self.obs_helper.get_section(obs, 'player_vel')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_vel = self.obs_helper.get_section(obs, 'opponent_vel')

        

        opp_pos[0] += opp_vel[0] * 1/30
        opp_pos[1] += opp_vel[1]* 1/30

        # if self.time % 10 == 0:
        #     print(opp_vel / 60)


        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()
        readible_obs = {
            "pos": obs[0:2],
            "vel": obs[2:4],
            "facing": (
                "right" 
                if self.obs_helper.get_section(obs, "player_facing")[0] > 0.5 
                else "left"
            ),
            "grounded": self.obs_helper.get_section(obs, "player_grounded")[0] > 0.5,
            "aerial": self.obs_helper.get_section(obs, "player_aerial")[0] > 0.5,
            "damage": self.obs_helper.get_section(obs, "player_damage")[0],
            "jumps_left": int(self.obs_helper.get_section(obs, "player_jumps_left")[0]),
            "opp_jumps_left": int(self.obs_helper.get_section(obs, "opponent_jumps_left")[0]),
            "stun_frames": self.obs_helper.get_section(obs, 'player_stun_frames')[0],
            "state": self.state_reverse.get(
                int(self.obs_helper.get_section(obs, 'player_state')[0]), 
                "UnknownState"
            ),
            "opp_state": self.state_reverse.get(
                int(self.obs_helper.get_section(obs, 'opponent_state')[0]), 
                "UnknownState"
            ),
            "weapon_type": int(self.obs_helper.get_section(obs, "player_weapon_type")[0]),
            "opp_weapon_type": int(self.obs_helper.get_section(obs, "opponent_weapon_type")[0]),
        }
        moving_platform = self.obs_helper.get_section(obs, "player_moving_platform_pos")
        moving_platform = [float(x) for x in moving_platform]
        all_platforms = self.platforms
        # all_platforms = self.platforms + [
        #     [moving_platform[0] - 0.2, moving_platform[0] + 0.2, moving_platform[1]]
        # ]
        keys = []
        targetX = 0
        targetY = 0
        MARGIN = 0.3

        spawner_list = [
            self.obs_helper.get_section(obs, "player_spawner_1"),
            self.obs_helper.get_section(obs, "player_spawner_2"),
            self.obs_helper.get_section(obs, "player_spawner_3"),
            self.obs_helper.get_section(obs, "player_spawner_4"),
        ]
        # return action
        # if self.time % 10 == 0:
        #     print(readible_obs["weapon_type"])
        # return action
        # Find nearest active spawner
        nearest_spawner = None
        nearest_distance = float('inf')

        for spawner in spawner_list:
            x, y, spawner_type = spawner

            if spawner_type == 0:  # inactive
                continue

            dx = x - pos[0]
            dy = y - pos[1]
            dist = math.sqrt(dx**2 + dy**2)

            if dist < nearest_distance:
                nearest_distance = dist
                nearest_spawner = spawner
        
        if nearest_spawner is not None and readible_obs["weapon_type"] == 0:
            targetX = nearest_spawner[0]
            targetY = nearest_spawner[1]
            if readible_obs["jumps_left"] >= 2 or (readible_obs["jumps_left"] == 0 and readible_obs["grounded"]):
                self.can_cross = True
            self.phase = "weapon_grab"

            if nearest_distance < 0.5:
                self.phase = "aggressive"
                keys.append("h")
                self.can_cross = False
        
        if nearest_spawner is None and readible_obs["weapon_type"] == 0 and readible_obs["opp_weapon_type"] != 0:
            self.phase = "flee"
            self.can_cross = True

        safe = False
        curr_platform = None
        for platform in all_platforms:
            if pos[0] > platform[0] and pos[0] < platform[1] and pos[1] < platform[2] + 0.2:
                safe = True
                curr_platform = platform
                self.last_platform = curr_platform

        super_safe = False
        for platform in all_platforms:
            if pos[0] > platform[0] + 1.5 and pos[0] < platform[1] - 1.5 and pos[1] < platform[2] + 0.2:
                super_safe = True


        opp_platform = None
        for platform in all_platforms:
            if opp_pos[0] > platform[0] and opp_pos[0] < platform[1] and opp_pos[1] < platform[2] + 0.2:
                opp_platform = platform
                
        dx = opp_pos[0] - pos[0]
        dy = opp_pos[1] - pos[1]
        distance_to_opp = math.sqrt(dx**2 + dy**2)

        # return action
        # GET THE SECTOR
        leeway = 0.3
        sector = None

        # vertical thresholds
        if abs(dx) <= leeway and dy < -leeway:
            sector = 2  # above
        elif abs(dx) <= leeway and dy > leeway:
            sector = 7  # below
        elif abs(dy) <= leeway and dx < -leeway:
            sector = 4  # left
        elif abs(dy) <= leeway and dx > leeway:
            sector = 5  # right
        elif dx < -leeway and dy < -leeway:
            sector = 1  # top-left
        elif dx > leeway and dy < -leeway:
            sector = 3  # top-right
        elif dx < -leeway and dy > leeway:
            sector = 6  # bottom-left
        elif dx > leeway and dy > leeway:
            sector = 8  # bottom-right
        else:
            sector = 0  # overlapping / same position
        # if self.time % 10 == 0:
        #     print(sector)

        # return action


        distances = []
        for platform in all_platforms:
            x_min, x_max, y_level = platform
            if pos[0] < x_min:
                dist = x_min - pos[0]
            elif pos[0] > x_max:
                dist = pos[0] - x_max
            else:
                dist = 0  # directly above platform

            if y_level < pos[1]:
                dist += (pos[1] - y_level) * 1.8

            distances.append(dist)

        # Find the nearest platform
        nearest_index = distances.index(min(distances))
        nearest_platform = all_platforms[nearest_index]
        x_min_nearest, x_max_nearest, y_level_nearest = nearest_platform
        do_not_move = False
        if dx > 0:
            opp_direction_x = "right"
        else:
            opp_direction_x = "left"

        if (curr_platform and opp_platform and curr_platform == opp_platform and self.phase == "passive") or self.damage_last_time < readible_obs["damage"]: #self.damage_last_time < readible_obs["damage"]
            self.phase = "aggressive"
            self.damage_last_time = readible_obs["damage"]
            self.can_cross = False

        if not safe and not self.can_cross:
            if pos[0] < x_min_nearest:
                targetX = x_min_nearest
            elif pos[0] > x_max_nearest:
                targetX = x_max_nearest

            targetY = y_level_nearest

        if self.phase == "passive" and safe:
            if opp_direction_x == "left":
                targetX = max(curr_platform[0], opp_pos[0])
            else:
                targetX = min(curr_platform[1], opp_pos[0])

            targetY = curr_platform[2]
        elif self.phase == "flee":
            flee_platform = all_platforms[0]
            if opp_platform is None:
                if curr_platform is not None:
                    flee_platform = curr_platform
            elif opp_platform == all_platforms[0]:
                flee_platform = all_platforms[1]
            elif opp_platform == all_platforms[0]:
                flee_platform = all_platforms[0]
            targetX = (flee_platform[1] + flee_platform[0]) / 2
            targetY = flee_platform[2]
        elif self.phase == "aggressive" and safe and not self.used_attack: # opp_direction_x == readible_obs["facing"]
            curr_weapon = readible_obs["weapon_type"]
            facing = readible_obs["facing"]
            grounded = readible_obs["grounded"]
            attack_state = "ground" if grounded else "aerial"
            
            curr_weapon_range = self.weapons_range[curr_weapon]
            if distance_to_opp < 1 + curr_weapon_range:
                attack_type = "close"
            elif distance_to_opp < 1.5 + curr_weapon_range:
                attack_type = "far"
            elif distance_to_opp < 2.2 + curr_weapon_range:
                attack_type = "lunge"
            else:
                attack_type = "out_of_range"

            if attack_type != "out_of_range":
                
                # Flip mapping for facing left
                sector_flip = {1: 3, 3: 1, 4: 5, 5: 4, 6: 8, 8: 6, 2: 2, 7: 7, 0: 0}
                eff_sector = sector_flip[sector] if facing == "left" else sector

                # Collect all valid moves
                valid_moves = []
                for move in self.weapons_data[curr_weapon]:
                    if move["type"] != attack_state:
                        continue
                    if move["range"] != attack_type:
                        continue
                    if eff_sector not in move["cover"]:
                        continue
                    if move.get("super_safe") and not super_safe:
                        continue
                    valid_moves.append(move)

                # Pick one at random if any valid moves exist
                if valid_moves:
                    move = random.choice(valid_moves)
                    keys_to_press = move["keys"].copy()

                    # Flip horizontal keys if facing left
                    if facing == "left":
                        keys_to_press = ["a" if k=="d" else "d" if k=="a" else k for k in keys_to_press]

                    # Press keys
                    for k in keys_to_press:
                        keys.append(k)
                    do_not_move = True
                    self.used_attack = True
        elif self.used_attack:
            self.used_attack = False

        # TRAVEL HERE
        if self.phase == "aggressive" and opp_platform is not None and self.last_platform is not None and self.last_platform != opp_platform \
            and not (readible_obs["aerial"] and readible_obs["jumps_left"] == 0 and pos[1] >= opp_platform[2]):
            targetX = (opp_platform[1] + opp_platform[0]) / 2
            targetY = opp_platform[2]
            
        elif not safe:
            MARGIN = 0
            self.can_cross = False
        elif self.phase == "aggressive" and not opp_KO:
            MARGIN = 0

           
            targetX = opp_pos[0]
            targetY = opp_pos[1]
            self.can_cross = False

        if not do_not_move:

            if pos[1] > moving_platform[1] and pos[0] > moving_platform[0] - 1.5 and pos[0] <  moving_platform[0] + 1.5:
                keys.append('a')
            elif pos[0] < targetX - MARGIN:
                keys.append('d')
            elif pos[0] >= targetX + MARGIN:
                keys.append('a')
        
        # print(player_vel)
        if pos[1] > targetY + MARGIN and self.time - self.lastJumped > 10:
            if not safe:
                self.lastJumped = self.time
                keys.append('space')
                if readible_obs["jumps_left"] == 0 and readible_obs["aerial"]:
                    keys.append('w')
                    keys.append('k')
            elif random.randint(0, 100) > 95:
                self.lastJumped = self.time
                keys.append('space')
        
        if distance_to_opp < 1 and readible_obs["opp_state"] == "AttackState" and self.time - self.last_dodged > 50:
            self.last_dodged = self.time
            keys = []
            keys.append("l")
            

        action = self.act_helper.press_keys(keys, action)
        return action


    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    # If modifying the number of models (or training in general), modify this
    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

        