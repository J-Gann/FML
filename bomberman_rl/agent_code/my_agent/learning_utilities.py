from sklearn import tree
from .path_utilities import move_to_nearest_coin
import events as e
import pickle

def setup_learning_features(self, load=True, save=True):
    if load:
        old_features_file = open('./data/old_features', 'rb')
        self.old_features = pickle.load(old_features_file)
        old_features_file.close()

        new_features_file = open('./data/new_features', 'rb')
        self.new_features = pickle.load(new_features_file)
        new_features_file.close()

        rewards_file = open('./data/rewards', 'rb')
        self.rewards = pickle.load(rewards_file)
        rewards_file.close()
    else:
        if not hasattr(self, "old_features"): self.old_features = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }
        if not hasattr(self, "new_features"): self.new_features = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }
        if not hasattr(self, "rewards"): self.rewards = { "UP": [], "DOWN": [], "LEFT": [], "RIGHT": [], "WAIT": [], "BOMB": [] }
    if save:
        old_features_file = open('./data/old_features', 'wb')
        pickle.dump(self.old_features, old_features_file)
        old_features_file.close()

        new_features_file = open('./data/new_features', 'wb')
        pickle.dump(self.new_features, new_features_file)
        new_features_file.close()

        rewards_file = open('./data/rewards', 'wb')
        pickle.dump(self.rewards, rewards_file)
        rewards_file.close()

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
