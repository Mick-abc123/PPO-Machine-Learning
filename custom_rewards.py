# Import CAR_MAX_SPEED from common game values
from abc import abstractmethod
from typing import Optional
from rlgym_sim.utils.common_values import CAR_MAX_SPEED, BACK_WALL_Y, BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED,  BALL_RADIUS, BACK_NET_Y
import numpy as np # Import numpy, the python math library
from rlgym_sim.utils import RewardFunction, math, common_values # Import the base RewardFunction class
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import FaceBallReward
from rlgym_sim.utils.gamestates import GameState, PlayerData # Import game state stuff

KPH_TO_VEL = 250/9
SIDE_WALL_X = 4096
SIDE_WALL_X2 = -4096
SIDE_WALL_BUFFER = 400
GOAL_HEIGHT = 642.775
ORANGE_GOAL_CENTER = (0, BACK_WALL_Y, GOAL_HEIGHT / 2)
BLUE_GOAL_CENTER = (0, -BACK_WALL_Y, GOAL_HEIGHT / 2)

def clamp(max_range: float, min_range: float, number: float) -> float:
    return max((min_range, min((number, max_range))))

def distance(x: np.array, y: np.array) -> float:
    return np.linalg.norm(x - y)


def distance2D(x: np.array, y: np.array)->float:
    x[2] = 0
    y[2] = 0
    return distance(x, y)

class SpeedflipKickoffReward(RewardFunction):
    def __init__(self, goal_speed=0.5):
        super().__init__()
        self.goal_speed = goal_speed

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.ball.position[0] == 0 and state.ball.position[1] == 0 and player.boost_amount < 2:
                vel = player.car_data.linear_velocity
                pos_diff = state.ball.position - player.car_data.position
                norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
                norm_vel = vel / CAR_MAX_SPEED
                speed_rew = self.goal_speed * max(float(np.dot(norm_pos_diff, norm_vel)), 0.025)
                return speed_rew
        return 0 
    
class VelocityReward(RewardFunction):
    # Simple reward function to ensure the model is training.
    def __init__(self, negative=False):
        super().__init__()
        self.negative = negative

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return np.linalg.norm(player.car_data.linear_velocity) / CAR_MAX_SPEED * (1 - 2 * self.negative)

class SpeedTowardBallReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Velocity of our player
        player_vel = player.car_data.linear_velocity
        
        # Difference in position between our player and the ball
        # When getting the change needed to reach B from A, we can use the formula: (B - A)
        pos_diff = (state.ball.position - player.car_data.position)
        
        # Determine the distance to the ball
        # The distance is just the length of pos_diff
        dist_to_ball = np.linalg.norm(pos_diff)
        
        # We will now normalize our pos_diff vector, so that it has a length/magnitude of 1
        # This will give us the direction to the ball, instead of the difference in position
        # Normalizing a vector can be done by dividing the vector by its length
        dir_to_ball = pos_diff / dist_to_ball

        # Use a dot product to determine how much of our velocity is in this direction
        # Note that this will go negative when we are going away from the ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)
        
        if speed_toward_ball > 0:
            # We are moving toward the ball at a speed of "speed_toward_ball"
            # The maximum speed we can move toward the ball is the maximum car speed
            # We want to return a reward from 0 to 1, so we need to divide our "speed_toward_ball" by the max player speed
            reward = speed_toward_ball / CAR_MAX_SPEED
            return reward
        else:
            # We are not moving toward the ball
            # Many good behaviors require moving away from the ball, so I highly recommend you don't punish moving away
            # We'll just not give any reward
            return 0
        
class VelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / BALL_MAX_SPEED
            return float(np.dot(norm_pos_diff, norm_vel))

class InAirReward(RewardFunction): # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
        # "player" is the current player we are getting the reward of
        # "state" is the current state of the game (ball, all players, etc.)
        # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array
        
        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0

class PlayerOnGroundReward(RewardFunction):

    def __init__(self):
        super().__init__()


    def reset(self, initial_state: GameState):
        pass 


    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
        if player.on_ground:
            # We are on ground, Return full reward
            return 1
        else:
            # We are in the air! No reward
            return 0

class TouchBallRewardScaledByHitForce(RewardFunction):
    def __init__(self):
        super().__init__()
        self.max_hit_speed = 130 * KPH_TO_VEL
        self.last_ball_vel = None
        self.cur_ball_vel = None

    # game reset, after terminal condition
    def reset(self, initial_state: GameState):
        self.last_ball_vel = initial_state.ball.linear_velocity
        self.cur_ball_vel = initial_state.ball.linear_velocity

    # happens 
    def pre_step(self, state: GameState):
        self.last_ball_vel = self.cur_ball_vel
        self.cur_ball_vel = state.ball.linear_velocity

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            reward = np.linalg.norm(self.cur_ball_vel - self.last_ball_vel) / self.max_hit_speed
            return reward
        return 0
    
RAMP_HEIGHT = 256

class AerialDistanceReward(RewardFunction):
    def __init__(self, height_scale: 1.1, distance_scale: 1.3):
        super().__init__()
        self.height_scale = height_scale
        self.distance_scale = distance_scale

        self.current_car: Optional[PlayerData] = None
        self.prev_state: Optional[GameState] = None
        self.ball_distance: float = 0
        self.car_distance: float = 0

    def reset(self, initial_state: GameState):
        self.current_car = None
        self.prev_state = initial_state

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = 0
        is_current = self.current_car is not None and self.current_car.car_id == player.car_id
        # Test if player is on the ground
        if player.car_data.position[2] < RAMP_HEIGHT:
            if is_current:
                is_current = False
                self.current_car = None
        # First non ground touch detection
        elif player.ball_touched and not is_current:
            is_current = True
            self.ball_distance = 0
            self.car_distance = 0
            rew = self.height_scale * max(player.car_data.position[2] + state.ball.position[2] - 2 * RAMP_HEIGHT, 0)
        # Still off the ground after a touch, add distance and reward for more touches
        elif is_current:
            self.car_distance += np.linalg.norm(player.car_data.position - self.current_car.car_data.position)
            self.ball_distance += np.linalg.norm(state.ball.position - self.prev_state.ball.position)
            # Cash out on touches
            if player.ball_touched:
                rew = self.distance_scale * (self.car_distance + self.ball_distance)
                self.car_distance = 0
                self.ball_distance = 0

        if is_current:
            self.current_car = player  # Update to get latest physics info

        self.prev_state = state

        return rew / (2 * BACK_WALL_Y)

class PlayerOnWallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return player.on_ground and player.car_data.position[2] > 300
    
class LavaFloorReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return -0.4 if player.on_ground else 0    
    
class OmniBoostDiscipline(RewardFunction):
    def __init__(self):
        super().__init__()
        self.values = [0 for _ in range(64)]

    def reset(self, initial_state: GameState):
        self.values = [0 for _ in range(64)]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        old, self.values[player.car_id] = self.values[player.car_id], player.boost_amount
        if player.car_data.position[2] < (2 * 92.75):
            return -int(self.values[player.car_id] < old)
        else:
            return 0

class JumpTouchReward(RewardFunction):
    def __init__(self, min_height=50):
        self.min_height = min_height
        self.max_height = 200
        self.range = self.max_height - self.min_height

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return (state.ball.position[2] - self.min_height) / self.range

        return 0


class AlignBallGoal(RewardFunction):
    def __init__(self, defense=1., offense=1.):
        super().__init__()
        self.defense = defense
        self.offense = offense

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc

        # Align player->ball and net->player vectors
        defensive_reward = self.defense * math.cosine_similarity(ball - pos, pos - protecc)

        # Align player->ball and player->net vectors
        offensive_reward = self.offense * math.cosine_similarity(ball - pos, attacc - pos)

        return defensive_reward + offensive_reward

class SaveBoostReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # 1 reward for each frame with 100 boost, sqrt because 0->20 makes bigger difference than 80->100
        return np.sqrt(player.boost_amount)

class BallYCoordinateReward(RewardFunction):
    def __init__(self, exponent=1):
        # Exponent should be odd so that negative y -> negative reward
        self.exponent = exponent

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            return (state.ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)) ** self.exponent
        else:
            return (state.inverted_ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)) ** self.exponent

class LiuDistanceBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False):
        super().__init__()
        self.own_goal = own_goal

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        # Compensate for moving objective to back of net
        dist = np.linalg.norm(state.ball.position - objective) - (BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS)
        return np.exp(-0.5 * dist / BALL_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

class TouchBallReward(RewardFunction):
    def __init__(self, aerial_weight=0.):
        self.aerial_weight = aerial_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            # Default just rewards 1, set aerial weight to reward more depending on ball height
            return ((state.ball.position[2] + BALL_RADIUS) / (2 * BALL_RADIUS)) ** self.aerial_weight
        return 0

class DribbleReward(RewardFunction):
    def __init__(self, speed_match_factor=2.0):
        super().__init__()
        self.MIN_BALL_HEIGHT = 109.0
        self.MAX_BALL_HEIGHT = 180.0
        self.MAX_DISTANCE = 197.0
        self.SPEED_MATCH_FACTOR = speed_match_factor
        self.PENALTY_RADIUS = 210
        self.PENALTY_VALUE = -1

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        car_pos = player.car_data.position
        car_vel = player.car_data.linear_velocity

        total_reward = 0.0
        
        # Check conditions for dribble reward
        if (player.on_ground and 
            self.MIN_BALL_HEIGHT <= ball_pos[2] <= self.MAX_BALL_HEIGHT and
            np.linalg.norm(car_pos - ball_pos) < self.MAX_DISTANCE):
            
            player_speed = np.linalg.norm(car_vel)
            ball_speed = np.linalg.norm(ball_vel)
            
            if player_speed + ball_speed > 0:  # Avoid division by zero
                speed_match_reward = (
                    (player_speed / common_values.CAR_MAX_SPEED) + 
                    self.SPEED_MATCH_FACTOR * (1.0 - abs(player_speed - ball_speed) / (player_speed + ball_speed))
                )
                total_reward += speed_match_reward

        # Penalty for multiple cars under the ball within the penalty radius
        for other_car in state.players:
            if other_car.car_id != player.car_id:  # Ignore the current player
                other_car_pos = other_car.car_data.position
                # Check if the other car is under the ball and within the penalty radius
                if np.linalg.norm(other_car_pos - ball_pos) < self.PENALTY_RADIUS and other_car_pos[2] < ball_pos[2]:
                    total_reward += self.PENALTY_VALUE
                    break  # Apply penalty once even if multiple cars are in the radius

        return total_reward


        
class AerialReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.BALL_MIN_HEIGHT = 300.0

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_pos = state.ball.position
        car_pos = player.car_data.position
        car_vel = player.car_data.linear_velocity
        
        # Compute direction to ball
        dir_to_ball = (ball_pos - car_pos) / np.linalg.norm(ball_pos - car_pos)
        
        # Compute distance to ball minus ball radius
        distance = np.linalg.norm(car_pos - ball_pos) - common_values.BALL_RADIUS
        distance_reward = np.exp(-0.5 * distance / common_values.CAR_MAX_SPEED)
        
        # Compute speed toward ball
        speed_toward_ball = np.dot(car_vel, dir_to_ball)
        speed_reward = speed_toward_ball / common_values.CAR_MAX_SPEED
        
        # Conditions
        if speed_toward_ball < 0:
            return 0.0
        if player.on_ground:
            return 0.0
        if ball_pos[2] < self.BALL_MIN_HEIGHT:
            return 0.0
        
        # Calculate reward
        reward = speed_reward * distance_reward
        
        #if player.has_double_jumped:
        #    reward += 0.1  # Double jump bonus
        
        return reward

class LightingMcQueenReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        car_vel = np.linalg.norm(player.car_data.linear_velocity)
        return np.sqrt(np.clip(car_vel / 1800, 0, 1))
    
class SlowPenalty(RewardFunction):
    def __init__(self, min_speed_threshold: float = 500, penalty: float = -1):
        super().__init__()
        self.min_speed_threshold = min_speed_threshold
        self.penalty = penalty

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        player_speed = np.linalg.norm(player.car_data.linear_velocity)
        if player_speed < self.min_speed_threshold:
            return self.penalty
        return 0.0


class AirDribbleReward(RewardFunction):
    def __init__(self, use_velocity_goal_reward=True, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.MIN_HEIGHT = 400
        self.MAX_HEIGHT = 1950
        self.MAX_DISTANCE = 250
        self.TOUCH_BONUS = 0  # Extra reward for touching the ball
        self.MOVE_AWAY_PENALTY = 0  # Penalty for moving away from the ball
        self.use_velocity_goal_reward = use_velocity_goal_reward
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_pos = state.ball.position
        car_pos = player.car_data.position
        car_vel = player.car_data.linear_velocity
        
        # Direction and speed towards the ball
        dir_to_ball = (ball_pos - car_pos) / np.linalg.norm(ball_pos - car_pos)
        speed_toward_ball = np.dot(car_vel, dir_to_ball)
        speed_reward = max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0)  # Ensure non-negative reward

        total_reward = 0

        # Check conditions for air dribble
        if (not player.on_ground and 
            self.MIN_HEIGHT <= ball_pos[2] <= self.MAX_HEIGHT and
            self.MIN_HEIGHT <= car_pos[2] <= self.MAX_HEIGHT and
            np.linalg.norm(car_pos - ball_pos) < self.MAX_DISTANCE):
            
            if speed_toward_ball > 0:
                total_reward += speed_reward
            else:
                total_reward += self.MOVE_AWAY_PENALTY  # Apply penalty if moving away from the ball
            
            # Add the extra reward if the player touches the ball
            if player.ball_touched:
                total_reward += self.TOUCH_BONUS
            
            # Velocity to goal reward integration
            if self.use_velocity_goal_reward:
                if player.team_num == BLUE_TEAM and not self.own_goal \
                        or player.team_num == ORANGE_TEAM and self.own_goal:
                    objective = np.array(ORANGE_GOAL_BACK)
                else:
                    objective = np.array(BLUE_GOAL_BACK)

                vel = state.ball.linear_velocity
                pos_diff = objective - ball_pos

                if self.use_scalar_projection:
                    # Scalar projection for velocity towards the goal
                    inv_t = math.scalar_projection(vel, pos_diff)
                    total_reward += inv_t
                else:
                    # Regular velocity component towards the goal
                    norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
                    norm_vel = vel / BALL_MAX_SPEED
                    goal_reward = float(np.dot(norm_pos_diff, norm_vel))
                    total_reward += goal_reward

        return total_reward


class AirRollReward(RewardFunction):
    def __init__(self, air_roll_w=1.0):
        super().__init__()
        self.air_roll_w = air_roll_w

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0

        # Check if the player car is within 1000 units of the ball
        car_to_ball_distance = np.linalg.norm(state.ball.position - player.car_data.position)
        if car_to_ball_distance < 1000:
            if not player.on_ground and player.car_data.position[2] > 4 * BALL_RADIUS and \
               not player.has_flip and player.car_data.forward()[2] > 0:
                car_to_ball = state.ball.position - player.car_data.position
                dot_product = np.dot(player.car_data.forward(), car_to_ball)
                if dot_product > 0:
                    reward += previous_action[4] * self.air_roll_w

        return reward

class SaveReward(RewardFunction):
    def __init__(self, protected_distance=1200, punish_area_entry=False, non_participation_reward=0.0):
        self.protected_distance = protected_distance
        self.punish_area_entry=punish_area_entry
        self.non_participation_reward = non_participation_reward
        self.needs_clear = False
        self.goal_spot = np.array([0, -5120, 0])

    def reset(self, initial_state: GameState):
        self.needs_clear = False

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball_loc = state.ball.position
        if player.team_num != 0:
            ball_loc = state.inverted_ball.position

        coord_diff = self.goal_spot - ball_loc
        ball_dist_2d = np.linalg.norm(coord_diff[:2])
        #ball_dist_2d = math.sqrt(coord_diff[0]*coord_diff[0] + coord_diff[1]*coord_diff[1])
        reward = 0

        if self.needs_clear:
            if ball_dist_2d > self.protected_distance:
                self.needs_clear = False
                if state.last_touch == player.car_id:
                    reward += 1
                else:
                    reward += self.non_participation_reward
        else:
            if ball_dist_2d < self.protected_distance:
                self.needs_clear = True
                if self.punish_area_entry:
                    reward -= 1
        return reward

class DefenderReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.enemy_goals = 0


    def reset(self, initial_state: GameState):
        pass

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.team_num == BLUE_TEAM:
            e_score = state.orange_score
            defend_loc = BLUE_GOAL_CENTER
        else:
            e_score = state.blue_score
            defend_loc = ORANGE_GOAL_CENTER

        if e_score > self.enemy_goals:
            self.enemy_goals = e_score
            dist = distance2D(np.asarray(defend_loc), player.car_data.position)
            if dist > 900:
                reward -= clamp(1, 0, dist/10000)
        return reward

class ChallengeReward(RewardFunction):
    def __init__(self, challenge_distance=300):
        super().__init__()
        self.challenge_distance = challenge_distance

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        # Check if the player is not on the ground and close to the ball
        if (
            not player.on_ground
            and distance(player.car_data.position, state.ball.position)
            < self.challenge_distance
        ):
            for _player in state.players:
                # Check if there are opponents close to the ball
                if (
                    _player.team_num != player.team_num
                    and distance(_player.car_data.position, state.ball.position)
                    < self.challenge_distance
                ):
                    reward += 0.1  # Basic reward for challenging with opponents nearby
                    if not player.has_flip:
                        # Additional reward if the player does not have a flip
                        reward += 0.9
                    break  # Reward once if at least one opponent is close

        return reward
    
class DribbleReward(RewardFunction):
    def __init__(self, challenge_distance=200):
        super().__init__()
        self.MIN_BALL_HEIGHT = 109
        self.MAX_BALL_HEIGHT = 180
        self.MAX_DISTANCE = 197
        self.SPEED_MATCH_FACTOR = 2
        self.flick_reward = 1.5
        self.previous_ball_vel = np.array([0, 0, 0])  # Initialize previous ball velocity
        self.challenge_distance = challenge_distance

    def reset(self, initial_state: GameState):
        self.previous_ball_vel = initial_state.ball.linear_velocity  # Reset previous velocity

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        car_pos = player.car_data.position
        car_vel = player.car_data.linear_velocity
        
        # Calculate the velocity change
        velocity_change = np.linalg.norm(ball_vel - self.previous_ball_vel)
        
        # Update previous velocity
        self.previous_ball_vel = ball_vel

        # Calculate speed match reward if dribbling correctly
        if (self.MIN_BALL_HEIGHT <= ball_pos[2] <= self.MAX_BALL_HEIGHT and
            np.linalg.norm(car_pos - ball_pos) < self.MAX_DISTANCE and
            -SIDE_WALL_X + SIDE_WALL_BUFFER < car_pos[0] < SIDE_WALL_X - SIDE_WALL_BUFFER):
                
            player_speed = np.linalg.norm(car_vel)
            ball_speed = np.linalg.norm(ball_vel)
            
            if player_speed + ball_speed > 0:  # Avoid division by zero
                speed_match_reward = (
                    (player_speed / common_values.CAR_MAX_SPEED) + 
                    self.SPEED_MATCH_FACTOR * (1.0 - abs(player_speed - ball_speed) / (player_speed + ball_speed))
                )
                reward += speed_match_reward
        
        return reward
    
class DribbleChallengeGroundPenalty(RewardFunction):
    def __init__(self, dribble_distance=150, challenge_distance=300, reward_amount=3, penalty_amount=-1):
        super().__init__()
        self.dribble_distance = dribble_distance  # Threshold distance to consider dribbling
        self.challenge_distance = challenge_distance  # Threshold distance to consider a challenge
        self.reward_amount = reward_amount  # Reward for a successful flick
        self.penalty_amount = penalty_amount  # Penalty for being on the ground during a challenge

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0
        ball_pos = state.ball.position
        car_pos = player.car_data.position
        ball_vel = state.ball.linear_velocity
        ball_dis = np.linalg.norm(car_pos - ball_pos)

        # Check if the player is dribbling (i.e., close to the ball)
        is_dribbling = distance(car_pos, ball_pos) < self.dribble_distance

        if is_dribbling:
            for opponent in state.players:
                if opponent.team_num != player.team_num:
                    opponent_pos = opponent.car_data.position
                    # Check if the opponent is close enough to be considered a challenge
                    if distance(opponent_pos, ball_pos) < self.challenge_distance:
                        # Penalize if the player is on the ground during a challenge
                        if player.on_ground:
                            reward += self.penalty_amount
                        # Reward if the player maintains control and attempts a flick
                        elif not player.on_ground and car_pos[2] < 120 and not player.has_flip:
                            reward += self.reward_amount

        return reward

class RetreatReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.defense_target = np.array(BLUE_GOAL_BACK)

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            ball = state.ball.position
            pos = player.car_data.position
            vel = player.car_data.linear_velocity
        else:
            ball = state.inverted_ball.position
            pos = player.inverted_car_data.position
            vel = player.inverted_car_data.linear_velocity

        reward = 0.0
        if ball[1]+200 < pos[1]:
            pos_diff = self.defense_target - pos
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / CAR_MAX_SPEED
            reward = np.dot(norm_pos_diff, norm_vel)
        return reward
    
class DemoPunish(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.demo_statuses = [True for _ in range(64)]

    def reset(self, initial_state: GameState) -> None:
        self.demo_statuses = [True for _ in range(64)]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        if player.is_demoed and not self.demo_statuses[player.car_id]:
            reward = -1

        self.demo_statuses[player.car_id] = player.is_demoed
        return reward
    
class AerialNavigation(RewardFunction):
    def __init__(
        self, ball_height_min=400, player_height_min=200, beginner=True
    ) -> None:
        super().__init__()
        self.ball_height_min = ball_height_min
        self.player_height_min = player_height_min
        self.face_reward = FaceBallReward()
        self.beginner = beginner

    def reset(self, initial_state: GameState) -> None:
        self.face_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if (
            not player.on_ground
            and state.ball.position[2] > self.ball_height_min
            and player.car_data.position[2] < self.ball_height_min
            and player.car_data.linear_velocity[2] > 0
            and distance2D(player.car_data.position, state.ball.position) < state.ball.position[2] * 3
        ):
            ball_direction = state.ball.position - player.car_data.position
            car_velocity = player.car_data.linear_velocity
            
            dot_product = np.dot(ball_direction, car_velocity)
            ball_direction_magnitude = np.linalg.norm(ball_direction)
            car_velocity_magnitude = np.linalg.norm(car_velocity)
            
            if ball_direction_magnitude > 0 and car_velocity_magnitude > 0:
                alignment = dot_product / (ball_direction_magnitude * car_velocity_magnitude)
            else:
                alignment = 0
            
            reward += alignment * 0.2  # Adjust scaling factor

        return max(reward, 0)

class Challange(RewardFunction):
    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.on_ground:
            for p in state.players:
                if (
                    p.car_id != player.car_id
                    and p.on_ground
                    and distance(player.car_data.position, p.car_data.position) < 300
                ):
                    return -1

        return 0

class GroundDribbleReward(RewardFunction): 
    def __init__(self):
        super().__init__()

        self.MIN_BALL_HEIGHT = 109.0
        self.MAX_BALL_HEIGHT = 180.0
        self.MAX_DISTANCE = 197.0
        self.COEFF = 2.0

    def reset(self, initial_state : GameState):
        pass

    def get_reward(self, player : PlayerData, state : GameState, previous_action):
        if player.on_ground and state.ball.position[2] >= self.MIN_BALL_HEIGHT \
            and state.ball.position[2] <= self.MAX_BALL_HEIGHT and np.linalg.norm(player.car_data.position - state.ball.position) < self.MAX_DISTANCE:
            
            player_speed = np.linalg.norm(player.car_data.linear_velocity)
            ball_speed = np.linalg.norm(state.ball.linear_velocity)
            player_speed_normalized = player_speed / CAR_MAX_SPEED
            inverse_difference = 1.0 - abs(player_speed - ball_speed)
            twosum = player_speed + ball_speed
            speed_reward = player_speed_normalized + self.COEFF * (inverse_difference / twosum)

            return speed_reward

        return 0.0 #originally was 0
    
class ZeroSumReward(RewardFunction):
    '''
    child_reward: The underlying reward function
    team_spirit: How much to share this reward with teammates (0-1)
    opp_scale: How to scale the penalty we get for the opponents getting this reward (usually 1)
    '''
    def __init__(self, child_reward: RewardFunction, team_spirit, opp_scale = 1.0):
        self.child_reward = child_reward # type: RewardFunction
        self.team_spirit = team_spirit
        self.opp_scale = opp_scale

        self._update_next = True
        self._rewards_cache = {}

    def reset(self, initial_state: GameState):
        self.child_reward.reset(initial_state)

    def pre_step(self, state: GameState):
        self.child_reward.pre_step(state)

        # Mark the next get_reward call as being the first reward call of the step
        self._update_next = True

    def update(self, state: GameState, is_final):
        self._rewards_cache = {}

        '''
        Each player's reward is calculated using this equation:
        reward = individual_rewards * (1-team_spirit) + avg_team_reward * team_spirit - avg_opp_reward * opp_scale
        '''

        # Get the individual rewards from each player while also adding them to that team's reward list
        individual_rewards = {}
        team_reward_lists = [ [], [] ]
        for player in state.players:
            if is_final:
                reward = self.child_reward.get_final_reward(player, state, None)
            else:
                reward = self.child_reward.get_reward(player, state, None)
            individual_rewards[player.car_id] = reward
            team_reward_lists[int(player.team_num)].append(reward)

        # If a team has no players, add a single 0 to their team rewards so the average doesn't break
        for i in range(2):
            if len(team_reward_lists[i]) == 0:
                team_reward_lists[i].append(0)

        # Turn the team-sorted reward lists into averages for each time
        # Example:
        #    Before: team_rewards = [ [1, 3], [4, 8] ]
        #    After:  team_rewards = [ 2, 6 ]
        team_rewards = np.average(team_reward_lists, 1)

        # Compute and cache:
        # reward = individual_rewards * (1-team_spirit)
        #          + avg_team_reward * team_spirit
        #          - avg_opp_reward * opp_scale
        for player in state.players:
            self._rewards_cache[player.car_id] = (
                    individual_rewards[player.car_id] * (1 - self.team_spirit)
                    + team_rewards[int(player.team_num)] * self.team_spirit
                    - team_rewards[1 - int(player.team_num)] * self.opp_scale
            )

    '''
    I made get_reward and get_final_reward both call get_reward_multi, using the "is_final" argument to distinguish
    Otherwise I would have to rewrite this function for final rewards, which is lame
    '''
    def get_reward_multi(self, player: PlayerData, state: GameState, previous_action: np.ndarray, is_final) -> float:
        # If this is the first get_reward call this step, we need to update the rewards for all players
        if self._update_next:
            self.update(state, is_final)
            self._update_next = False
        return self._rewards_cache[player.car_id]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward_multi(player, state, previous_action, False)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward_multi(player, state, previous_action, True)
