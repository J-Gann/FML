from sklearn import tree
from .path_utilities import move_to_nearest_coin
import events as e

def setup_learning_features(self, load=True, save=True):
    self.old_features = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }
    self.new_features = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }
    self.rewards = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }

def update_transitions(self, old_game_state, self_action, new_game_state, events):
    old_features = features_from_game_state(self, old_game_state, self_action)
    new_features = features_from_game_state(self, new_game_state, self_action)
    rewards = _rewards_from_events(events)
    self.old_features[self_action].append(old_features)
    self.new_features[self_action].append(new_features)
    self.rewards[self_action].append(rewards)

def features_from_game_state(self, game_state, self_action):
    move = move_to_nearest_coin(self, game_state["self"][3], game_state["coins"])
    if move == self_action: return [1]
    else: return [0]

def _rewards_from_events(events):
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum