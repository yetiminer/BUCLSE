import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


class Encoder(nn.Module):
	''' This the encoder part of VAE

	'''
	def __init__(self, input_dim=None, hidden_dim=None, latent_dim=2, cond_dim=None):
		'''
		Args:
			input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
			hidden_dim: A integer indicating the size of hidden dimension.
			latent_dim: A integer indicating the latent size.
			n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
		'''
		super().__init__()

		self.linear = nn.Linear(input_dim+cond_dim, hidden_dim)
		self.mu = nn.Linear(hidden_dim, latent_dim)
		self.var = nn.Linear(hidden_dim, latent_dim)

	def forward(self, x):
		# x is of shape [batch_size, input_dim + n_classes]

		hidden = torch.relu(self.linear(x))
		# hidden is of shape [batch_size, hidden_dim]

		# latent parameters
		mean = self.mu(hidden)
		# mean is of shape [batch_size, latent_dim]
		log_var = self.var(hidden)
		# log_var is of shape [batch_size, latent_dim]

		return mean, log_var


class Decoder(nn.Module):
	''' This the decoder part of VAE - moving from latent dimension to data output

	'''
	def __init__(self, latent_dim=2, hidden_dim=None, output_dim=None, cond_dim=None,done=None,reward=None):
		'''
		Args:
			latent_dim: A integer indicating the latent size.
			hidden_dim: A integer indicating the size of hidden dimension.
			output_dim: A integer indicating the size of output (in case of MNIST 28 * 28).
			cond_dim: A integer indicating the dimension of conditioning variables
		'''
		super().__init__()

		self.latent_to_hidden = nn.Linear(latent_dim + cond_dim, hidden_dim)
		self.hidden_to_out = nn.Linear(hidden_dim, output_dim)
		
		self.reward_layer=False
		self.done_layer=False
		
		if reward is None:
			self.reward=nn.Linear(output_dim,1)
			self_reward_layer=True
		else:
			self.reward=reward
		if done is None:
			self.done=nn.Linear(output_dim,1)
			self.done_layer=True
		else:
			self.done=done
		
		

	def forward(self, x):
		# x is of shape [batch_size, latent_dim + num_classes]
		x = torch.relu(self.latent_to_hidden(x))
		# x is of shape [batch_size, hidden_dim]
		state_prime = torch.relu(self.hidden_to_out(x))
		# x is of shape [batch_size, output_dim]
		reward_value=self.reward(state_prime)
		if self.reward_layer:
			reward_value=torch.relu(reward_value)

		done_value=self.done(state_prime)
		if self.done_layer:
			done_value=torch.relu(done_value)

		return state_prime, reward_value,done_value
		
class CVAE(nn.Module):
	''' This the VAE, which takes a encoder and decoder.

	'''

	def __init__(self,N_STATES,N_ACTIONS,H1Size,H2Size,latent_dim=2,done=None,reward=None):
		'''
		Args:
			input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
			hidden_dim: A integer indicating the size of hidden dimension.
			latent_dim: A integer indicating the latent size.
			n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
		'''
		super().__init__()
		
		input_dim=N_STATES
		hidden_dim=H1Size
		cond_dim=N_STATES+N_ACTIONS
		output_dim=input_dim #because we're estimating reward and done
		
		self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, cond_dim=cond_dim)
		self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim, cond_dim=cond_dim,done=done,reward=reward)
		
		self.reward_layer=self.decoder.reward_layer
		self.done_layer=self.decoder.done_layer

	def forward(self, x, y):

		x = torch.cat((x, y), dim=1)

		# encode
		z_mu, z_var = self.encoder(x)

		# sample from the distribution having latent parameters z_mu, z_var
		# reparameterize
		std = torch.exp(z_var / 2)
		eps = torch.randn_like(std)
		x_sample = eps.mul(std).add_(z_mu)

		z = torch.cat((x_sample, y), dim=1)

		# decode
		generated_x,reward_value,done_value = self.decoder(z)

		return generated_x,reward_value,done_value, z_mu, z_var
		
def CVAE_loss_make(weight):
	def CVAE_loss(x, reconstructed_x, mean, log_var,recon_loss):
		# reconstruction loss
		RCL = recon_loss(reconstructed_x, x)
		# kl divergence loss
		KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

		return weight*RCL + KLD
		
	def CVAE_loss_parts(x, reconstructed_x, mean, log_var,recon_loss):
		# reconstruction loss
		RCL = recon_loss(reconstructed_x, x)
		# kl divergence loss
		KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

		return weight*RCL, KLD
		
	return CVAE_loss, CVAE_loss_parts