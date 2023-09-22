import torch
import torch.nn as nn
from skrl.models.torch import Model, DeterministicMixin, GaussianMixin

class PolicyCNN(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        # shared layers/network
        self.features_extractor = nn.Sequential(nn.Conv1d(1, 32, kernel_size=5, stride=3),
                                                nn.ReLU(),
                                                nn.Conv1d(32, 32, kernel_size=3, stride=2),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                )
        # Shape = batch, channel, number of lidar readings
        sample = torch.Tensor(observation_space.sample()['lidar'])
        sample = sample.reshape(-1, 1, observation_space['lidar'].shape[-1])
        nflatten = self.features_extractor(sample).shape[-1]

        self.net = nn.Sequential(nn.Linear(nflatten + observation_space['odom'].shape[-1], 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, self.num_actions),
                                 nn.Tanh()
                                 )
        
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    # forward the input to compute model output according to the specified role
    def compute(self, inputs, role):
        raw = inputs['states']
        data_size=[self.observation_space[k].shape[-1]for k in self.observation_space.keys()]
        data = {}
        start = 0 
        for i, k in enumerate(self.observation_space.keys()):
            data[k] = raw[:, start:start+data_size[i]]
            start = data_size[i]
        x = self.features_extractor(data['lidar'].reshape(-1,1,self.observation_space['lidar'].shape[-1]))
        x = torch.cat([x, data['odom'].reshape(-1, self.observation_space['odom'].shape[-1])], dim=1)
        x = self.net(x)
        
        return x, self.log_std_parameter, {}
    
class ValueCNN(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.features_extractor = nn.Sequential(nn.Conv1d(1, 32, kernel_size=5, stride=3),
                                                nn.ReLU(),
                                                nn.Conv1d(32, 32, kernel_size=3, stride=2),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                )
        # Shape = batch, channel, number of lidar readings
        sample = torch.Tensor(observation_space.sample()['lidar'])
        sample = sample.reshape(-1, 1, observation_space['lidar'].shape[-1])
        nflatten = self.features_extractor(sample).shape[-1]

        self.net = nn.Sequential(nn.Linear(nflatten + observation_space['odom'].shape[-1], 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 1),
                                 )

    def compute(self, inputs, role):
        raw = inputs['states']
        data_size=[self.observation_space[k].shape[-1]for k in self.observation_space.keys()]
        data = {}
        start = 0 
        for i, k in enumerate(self.observation_space.keys()):
            data[k] = raw[:, start:start+data_size[i]]
            start = data_size[i]

        x = self.features_extractor(data['lidar'].reshape(-1,1,self.observation_space['lidar'].shape[-1]))
        x = torch.cat([x, data['odom'].reshape(-1, self.observation_space['odom'].shape[-1])], dim=1)
        return self.net(x), {}