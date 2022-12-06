from torch import nn
from torch.distributions import normal
import torch

class HeadNet(nn.Module):
    def __init__(self,input,output):
        super(HeadNet, self).__init__()
        self.fc2 = nn.Linear(input, output)
        self.soft = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.soft(self.fc2(x))
        return x


class CoreNet(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(CoreNet, self).__init__()
		self.tanh1 = nn.Tanh()
		self.lin1 = nn.Linear(input_size, hidden_size)
		self.tanh2 = nn.Tanh()
		self.lin2 = nn.Linear(hidden_size, hidden_size//2)
	def forward(self, x):
		x = self.lin1(x)
		x = self.tanh1(x)
		x = self.lin2(x)
		x = self.tanh2(x)
		return x
	

class Actor(nn.Module):
    def __init__(self, n_states, n_actions,bins,heads = 1):
        super(Actor, self).__init__()
        hidden_size =  100
        hidden_size_2 =  50
        self.core_net = CoreNet(n_states, hidden_size)
        self.heads = heads
        self.head0 = nn.ModuleList([HeadNet(hidden_size_2,bins) for action in range(n_actions)])
        if self.heads>=5:
            self.head1 = nn.ModuleList([HeadNet(hidden_size_2,bins) for action in range(n_actions)])
            self.head2 = nn.ModuleList([HeadNet(hidden_size_2,bins) for action in range(n_actions)])
            self.head3 = nn.ModuleList([HeadNet(hidden_size_2,bins) for action in range(n_actions)])
            self.head4 = nn.ModuleList([HeadNet(hidden_size_2,bins) for action in range(n_actions)])
        if self.heads==10:
            self.head5 = nn.ModuleList([HeadNet(hidden_size_2,bins) for action in range(n_actions)])
            self.head6 = nn.ModuleList([HeadNet(hidden_size_2,bins) for action in range(n_actions)])
            self.head7 = nn.ModuleList([HeadNet(hidden_size_2,bins) for action in range(n_actions)])
            self.head8 = nn.ModuleList([HeadNet(hidden_size_2,bins) for action in range(n_actions)])
            self.head9 = nn.ModuleList([HeadNet(hidden_size_2,bins) for action in range(n_actions)])

    #     self.head0.apply(self._init_weights)
        
    #     for layer in self.modules():
    #         if isinstance(layer, nn.Linear):
    #             nn.init.orthogonal_(layer.weight)
    #             layer.bias.data.zero_()
		
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.orthogonal_(m.weight)
    #         m.bias.data.zero_()

    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x ):
        if self.heads == 1:
            return {"head0" : [net(x) for net in self.head0]}
        elif self.heads == 5:
            return {"head0" : [net(x) for net in self.head0],
                    "head1" : [net(x) for net in self.head1],
                    "head2" : [net(x) for net in self.head2],
                    "head3" : [net(x) for net in self.head3],
                    "head4" : [net(x) for net in self.head4]}
        elif self.heads == 10:
            return {"head0" : [net(x) for net in self.head0],
                    "head1" : [net(x) for net in self.head1],
                    "head2" : [net(x) for net in self.head2],
                    "head3" : [net(x) for net in self.head3],
                    "head4" : [net(x) for net in self.head4],
                    "head5" : [net(x) for net in self.head5],
                    "head6" : [net(x) for net in self.head6],
                    "head7" : [net(x) for net in self.head7],
                    "head8" : [net(x) for net in self.head8],
                    "head9" : [net(x) for net in self.head9]}

    def forward(self, x):
        core_cache =self._core(x)
        net_heads = self._heads(core_cache)
        return net_heads

class Critic(nn.Module):
    def __init__(self, n_states):
        super(Critic, self).__init__()
        self.n_states = n_states

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.value = nn.Linear(in_features=64, out_features=1)

        # for layer in self.modules():
        #     if isinstance(layer, nn.Linear):
        #         nn.init.orthogonal_(layer.weight)
        #         layer.bias.data.zero_()

    def forward(self, inputs):
        x = inputs
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.value(x)

        return value
