import torch as th

class Network(th.nn.Module):


    def __init__(self):
        super(Network, self).__init__()

        self.possible_actions = th.FloatTensor([[1,0,0], [-1,0,0], [0,1,0], [0,0,0.8], [0,0,0]])

    def get_network_prediction(self, current_obs, observations, action_indices):
        # TODO


        action_idx = th.randint(0, len(self.possible_actions), (1,)).item()
        action = self.possible_actions[action_idx].tolist()

        return action, action_idx

    def convert_simulation_data_to_buffer_items(self, observations, action_indices, rewards):
        items = th.randint(0, 10, (len(observations),)).tolist()

        # TODO

        return items

    def forward(self, x):
        # TODO
        idx = th.randint(0, len(self.possible_actions), (1,)).item()
        return self.possible_actions[idx], idx

    def do_train_step(self):
        raise NotImplementedError()