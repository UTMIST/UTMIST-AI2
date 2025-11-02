# # SUBMISSION: Competitive Rule-Based Agent
# This is a highly competitive if-else agent that doesn't use any ML.
# Strategy: Aggressive spacing, smart attack selection, edge guarding, and defensive play

import os
import numpy as np
from typing import Optional
from environment.agent import Agent

# To run the sample TTNN model, you can uncomment the 2 lines below: 
# import ttnn
# from user.my_agent_tt import TTMLPPolicy


class SubmittedAgent(Agent):
    '''
    Competitive Rule-Based Fighting Agent
    
    Strategy:
    1. Positioning: Maintain center stage control, avoid edges
    2. Spacing: Keep optimal attack distance from opponent
    3. Attack Selection: Use fast attacks close range, heavy attacks at medium range
    4. Defense: Dodge when opponent attacks, punish recovery frames
    5. Edge Guarding: Pressure opponent when near edges
    6. Weapon Management: Pick up weapons when safe
    '''
    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        super().__init__(file_path)
        
        # Internal state tracking
        self.frame_counter = 0
        self.last_action_frame = 0
        self.action_cooldown = 0
        self.dodge_cooldown = 0
        self.last_distance = 0
        self.consecutive_hits = 0
        self.last_opponent_damage = 0
        
        # Stage boundaries (from observation space analysis)
        self.stage_left = -0.9  # Normalized coordinates
        self.stage_right = 0.9
        self.stage_center = 0.0
        self.edge_danger_zone = 0.15  # Distance from edge to consider dangerous
        
        # Combat parameters
        self.optimal_close_range = 0.15  # Very close for light attacks
        self.optimal_mid_range = 0.30    # Medium for heavy attacks
        self.retreat_distance = 0.50     # Too far, need to approach
        self.attack_cooldown_frames = 15 # Frames between attacks

        # To run a TTNN model, you must maintain a pointer to the device and can be done by 
        # uncommmenting the line below to use the device pointer
        # self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))

    def _initialize(self) -> None:
        """Initialize agent - no model needed for rule-based"""
        pass

        # To run the sample TTNN model during inference, you can uncomment the 5 lines below:
        # This assumes that your self.model.policy has the MLPPolicy architecture defined in `train_agent.py` or `my_agent_tt.py`
        # mlp_state_dict = self.model.policy.features_extractor.model.state_dict()
        # self.tt_model = TTMLPPolicy(mlp_state_dict, self.mesh_device)
        # self.model.policy.features_extractor.model = self.tt_model
        # self.model.policy.vf_features_extractor.model = self.tt_model
        # self.model.policy.pi_features_extractor.model = self.tt_model

    def _gdown(self) -> str:
        """No file download needed"""
        return None

    def predict(self, obs):
        """
        Main decision-making logic
        
        Observation breakdown (64 dims):
        0-1: player_pos (x, y)
        2-3: player_vel (vx, vy)
        4: player_facing
        5: player_grounded
        6: player_aerial
        7: player_jumps_left
        8: player_state
        9: player_recoveries_left
        10: player_dodge_timer
        11: player_stun_frames
        12: player_damage
        13: player_stocks
        14: player_move_type
        15: player_weapon_type
        16-27: player_spawners (4x 3dims: x,y,type)
        28-29: player_platform_pos
        30-31: player_platform_vel
        32-63: opponent_ (same structure)
        """
        self.frame_counter += 1
        self.action_cooldown = max(0, self.action_cooldown - 1)
        self.dodge_cooldown = max(0, self.dodge_cooldown - 1)
        
        # Parse observation
        my_x, my_y = obs[0], obs[1]
        my_vx, my_vy = obs[2], obs[3]
        my_facing = obs[4]
        my_grounded = obs[5] > 0.5
        my_aerial = obs[6] > 0.5
        my_jumps_left = int(obs[7])
        my_damage = obs[12]
        my_weapon = int(obs[15])
        
        opp_x, opp_y = obs[32], obs[33]
        opp_vx, opp_vy = obs[34], obs[35]
        opp_grounded = obs[37] > 0.5
        opp_stun_frames = obs[43]
        opp_damage = obs[44]
        opp_move_type = int(obs[46])
        
        # Calculate key metrics
        distance_x = opp_x - my_x
        distance_y = opp_y - my_y
        distance = np.sqrt(distance_x**2 + distance_y**2)
        
        my_distance_from_center = abs(my_x)
        opp_distance_from_center = abs(opp_x)
        
        my_near_left_edge = (my_x < -self.stage_left + self.edge_danger_zone)
        my_near_right_edge = (my_x > self.stage_right - self.edge_danger_zone)
        my_near_edge = my_near_left_edge or my_near_right_edge
        
        opp_near_left_edge = (opp_x < -self.stage_left + self.edge_danger_zone)
        opp_near_right_edge = (opp_x > self.stage_right - self.edge_danger_zone)
        opp_near_edge = opp_near_left_edge or opp_near_right_edge
        
        # Track if we hit opponent (damage increased)
        if opp_damage > self.last_opponent_damage:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0
        self.last_opponent_damage = opp_damage
        
        # Initialize action (all keys released)
        action = self.act_helper.zeros()
        
        # === PRIORITY 1: EMERGENCY SURVIVAL ===
        # If we're near edge and in danger, recover to center immediately
        if my_near_edge and (my_aerial or not my_grounded):
            if my_near_left_edge:
                action = self.act_helper.press_keys(['d', 'space'])  # Right + Jump
            else:
                action = self.act_helper.press_keys(['a', 'space'])  # Left + Jump
            return action
        
        # If stunned or being hit, try to dodge when we recover
        if opp_stun_frames > 0 or self.dodge_cooldown == 0:
            if distance < 0.3 and opp_move_type > 0 and self.dodge_cooldown == 0:  # Opponent attacking
                # Dodge away
                dodge_direction = 'a' if distance_x > 0 else 'd'
                action = self.act_helper.press_keys([dodge_direction, 'l'])
                self.dodge_cooldown = 60
                return action
        
        # === PRIORITY 2: STAGE CONTROL ===
        # Maintain center stage advantage
        if not my_near_edge and my_distance_from_center > 0.3:
            # Move toward center
            if my_x < 0:
                action = self.act_helper.press_keys('d')  # Move right toward center
            else:
                action = self.act_helper.press_keys('a')  # Move left toward center
            
            # Jump if grounded and need vertical adjustment
            if my_grounded and abs(distance_y) > 0.2:
                action = self.act_helper.press_keys(['space'] + list(self.act_helper.sections.keys())[:4])
            return action
        
        # === PRIORITY 3: WEAPON PICKUP ===
        # Pick up weapon if we don't have one and it's safe
        if my_weapon == 0 and distance > 0.4 and my_grounded:
            # Check spawners for nearby weapons
            for i in range(4):
                spawner_x = obs[16 + i*3]
                spawner_y = obs[17 + i*3]
                spawner_type = obs[18 + i*3]
                if spawner_type > 0:  # Weapon exists
                    weapon_distance = np.sqrt((spawner_x - my_x)**2 + (spawner_y - my_y)**2)
                    if weapon_distance < 0.3:
                        action = self.act_helper.press_keys('h')  # Pickup
                        return action
        
        # === PRIORITY 4: EDGE GUARDING ===
        # If opponent is near edge and we have advantage, pressure them
        if opp_near_edge and opp_distance_from_center > my_distance_from_center:
            # Position between opponent and center
            target_x = opp_x + (0.2 if opp_near_right_edge else -0.2)
            
            if abs(my_x - target_x) > 0.1:
                if my_x < target_x:
                    action = self.act_helper.press_keys('d')
                else:
                    action = self.act_helper.press_keys('a')
            
            # Attack if in range
            if distance < self.optimal_mid_range and self.action_cooldown == 0:
                if my_aerial:
                    action = self.act_helper.press_keys(['s', 'j'])  # Down air
                else:
                    action = self.act_helper.press_keys('k')  # Heavy attack
                self.action_cooldown = self.attack_cooldown_frames
            
            return action
        
        # === PRIORITY 5: COMBAT - SPACING & ATTACKING ===
        
        # Calculate if we're facing opponent
        facing_opponent = (my_facing > 0.5 and distance_x > 0) or (my_facing < 0.5 and distance_x < 0)
        
        # Opponent is stunned - PUNISH!
        if opp_stun_frames > 5 and distance < self.optimal_mid_range:
            if my_aerial:
                # Aerial combo
                if distance < self.optimal_close_range:
                    action = self.act_helper.press_keys('j')  # Aerial light
                else:
                    action = self.act_helper.press_keys(['s', 'j'])  # Down air
            else:
                # Ground combo
                if my_weapon > 0:
                    action = self.act_helper.press_keys('k')  # Heavy with weapon
                else:
                    action = self.act_helper.press_keys('j')  # Light attack
            
            self.action_cooldown = 10  # Shorter cooldown for combos
            return action
        
        # Spacing logic based on distance
        if distance < self.optimal_close_range:
            # TOO CLOSE - Light attack or create space
            if self.action_cooldown == 0:
                if my_aerial:
                    action = self.act_helper.press_keys('j')  # Neutral air
                else:
                    action = self.act_helper.press_keys('j')  # Light attack
                self.action_cooldown = self.attack_cooldown_frames
            else:
                # Create space
                retreat_dir = 'a' if distance_x > 0 else 'd'
                action = self.act_helper.press_keys(retreat_dir)
        
        elif distance < self.optimal_mid_range:
            # OPTIMAL RANGE - Heavy attacks
            if self.action_cooldown == 0:
                if my_aerial:
                    # Aerial heavy or directional aerial
                    if distance_y > 0.1:
                        action = self.act_helper.press_keys(['w', 'j'])  # Up air
                    elif distance_y < -0.1:
                        action = self.act_helper.press_keys(['s', 'j'])  # Down air
                    else:
                        action = self.act_helper.press_keys('k')  # Aerial heavy
                else:
                    # Ground heavy
                    action = self.act_helper.press_keys('k')
                self.action_cooldown = self.attack_cooldown_frames
            else:
                # Maintain spacing
                if not facing_opponent:
                    # Turn around
                    turn_dir = 'd' if distance_x > 0 else 'a'
                    action = self.act_helper.press_keys(turn_dir)
        
        elif distance < self.retreat_distance:
            # MID RANGE - Approach carefully or jump in
            if my_grounded and my_jumps_left > 0 and distance < 0.4:
                # Jump in for aerial approach
                approach_dir = 'd' if distance_x > 0 else 'a'
                action = self.act_helper.press_keys([approach_dir, 'space'])
            else:
                # Ground approach
                approach_dir = 'd' if distance_x > 0 else 'a'
                action = self.act_helper.press_keys(approach_dir)
        
        else:
            # FAR RANGE - Close distance quickly
            approach_dir = 'd' if distance_x > 0 else 'a'
            if my_grounded:
                # Sprint toward opponent
                action = self.act_helper.press_keys([approach_dir, approach_dir])  # Double tap for dash
            else:
                action = self.act_helper.press_keys(approach_dir)
        
        self.last_distance = distance
        return action

    def save(self, file_path: str) -> None:
        """No model to save"""
        pass

    def learn(self, env, total_timesteps, log_interval: int = 4):
        """No learning for rule-based agent"""
        pass
