#fake cash

from UCLSE.dyna_q.Experiment1a import SimpleRLEnv_mod,Experiment, GetOutOfLoop, ProfitWeird, LossRecord
import random
from UCLSE.dyna_q.dyna_q import DynaQ
from UCLSE.wang_wellman_new import EnvFactory
import UCLSE.dyna_q.utils as utils
import numpy as np
import warnings
import time

		
class Experiment(Experiment):

		
		def initiate(self,trader_pref_kwargs=None,timer_kwargs=None,price_sequence_kwargs=None,noise_kwargs=None,
		messenger_kwargs=None,env_kwargs=None,trader_kwargs=None,lobenv_kwargs=None,dyna_kwargs=None,agent_kwargs=None,memory=None,tabular=None,agent=DynaQ,name='experiment_no_name'):
			#needed this for loading from checkpoint
			self.name=name
			self.trader_pref_kwargs=trader_pref_kwargs
			self.timer_kwargs=timer_kwargs
			self.price_sequence_kwargs=price_sequence_kwargs
			self.noise_kwargs=noise_kwargs
			self.messenger_kwargs=messenger_kwargs
			self.env_kwargs=env_kwargs
			self.trader_kwargs=trader_kwargs
			self.lobenv_kwargs=lobenv_kwargs
			self.dyna_kwargs=dyna_kwargs
			self.agent_kwargs=agent_kwargs
			self.agent_type=agent
			
			EF1,EF2,EF3=(EnvFactory(trader_pref_kwargs=trader_pref_kwargs,
					timer_kwargs=timer_kwargs,price_sequence_kwargs=price_sequence_kwargs,
					noise_kwargs=noise_kwargs,trader_kwargs=trader_kwargs,env_kwargs=env_kwargs,
					messenger_kwargs=messenger_kwargs,name=str(k)) for k in range(3))
			
			self.EF_test=EnvFactory(trader_pref_kwargs=trader_pref_kwargs,
					timer_kwargs=timer_kwargs,price_sequence_kwargs=price_sequence_kwargs,
					noise_kwargs=noise_kwargs,trader_kwargs=trader_kwargs,env_kwargs=env_kwargs,
					messenger_kwargs=messenger_kwargs)
			
			
			self.cutoff=lobenv_kwargs['cutoff']
			self.profit_target=lobenv_kwargs['profit_target']
			self.loss_limit=lobenv_kwargs['loss_limit']
			
						  
			lobenv1=SimpleRLEnv_mod.setup(EnvFactory=EF1,parent=SimpleRLEnv_mod,**lobenv_kwargs)
			lobenv2=SimpleRLEnv_mod.setup(EnvFactory=EF2,parent=SimpleRLEnv_mod,**lobenv_kwargs)
			lobenv3=SimpleRLEnv_mod.setup(EnvFactory=EF3,parent=SimpleRLEnv_mod,**lobenv_kwargs)
			self.lobenvs=[lobenv1,lobenv2,lobenv3]
			
			
			self.dyna_config=None
			if dyna_kwargs is not None:
				self.dyna_config=self.setup_dyna_config(dyna_kwargs)
			
			if self.dyna_config is not None:
				self.agent=agent(self.dyna_config,**agent_kwargs)
			else:
				self.agent=agent(**agent_kwargs)
				
		def _train_setup(self,MaxEpisodes=100,planning_steps=5,lookback=50,thresh=5,planning=True,graph=False,epsilon=None,
							total_steps=0,episode=0,novel_list=[],rwd_dyna=[],best_rew=(0,0,0),rwd_dyna_test=[]):
				self.MaxEpisodes=MaxEpisodes
				self.planning_steps=planning_steps #number of planning sweeps
				self.exp=0
				self.lookback=lookback
				self.thresh=thresh
				self.planning=planning
				if epsilon is None:
					self.EPSILON = self.dyna_config['exploration']['init_epsilon']
				else:
					self.EPSILON=epsilon
				
				self.graph=graph
				if graph and self.vis is not None:
					if graph:
						self._setup_graph()
						
				self.total_steps = total_steps
				self.episode=episode
				self.novel_list=novel_list
				self.rwd_dyna =rwd_dyna
				self.rwd_dyna_test =rwd_dyna_test
				self.best_rew=best_rew
				self.test_counter=0
				
				
		def _setup_graph(self):
				self.train_loss_window = self.__create_plot_window(self.vis, '#Iterations', 'Loss', self.name + ': Training Loss')
				time.sleep(0.5)
				self.vis.get_window_data(self.train_loss_window)
				self.train_return_hist=self.__create_bar_window(self.vis, self.name+': Return distribution')
				time.sleep(0.5)
				self.vis.get_window_data(self.train_return_hist)
				self.state_window=self.__create_plot_window(self.vis, 'Episode #', '#states',self.name+ ': States explored')
				time.sleep(0.5)
				self.vis.get_window_data(self.state_window)
				self.test_loss_window = self.__create_plot_window(self.vis, '#Iterations', 'Loss', self.name + ': Test Loss')
				time.sleep(0.5)
				
				assert self.train_loss_window!=self.train_return_hist!=self.state_window!=self.test_loss_window
				

		def train(self,MaxEpisodes=100,start_episode=0,total_steps=0,folder=None):
			if folder is None:
				print('Specify path to save checkpoints')
				raise AttributeError
			else:
				self.folder=folder
				
			print(f'Planning is {self.planning}, double Q model is {self.dyna_config["double_q_model"]}, tabular memory is {self.dyna_config["memory"]["tabular memory"]}')
			self.temp_explo_data=[]
			self.best_counter=0 #this is a counter that increments
			
			discount=self.dyna_config['discount']
			self.last_test=start_episode
			
			try: 

			
				for i_episode in range(start_episode,MaxEpisodes):
					
					#select a new environment
					lobenv=self.env_selector(i_episode,self.lobenvs)
					start_balance=lobenv.trader.balance
					ep_r = 0
					timestep = 0
					s,r0 = lobenv.reset()
					#fc=lobenv.set_fake_cash()
					initial=True
					ep_r=lobenv.lamb*r0
					
					
					while True:
						total_steps += 1
						timestep += 1
						self.total_steps=total_steps

						# decay exploration
						self.EPSILON = utils.epsilon_decay(
							eps=self.EPSILON, 
							step=self.total_steps, 
							config=self.dyna_config['exploration']
						)
						assert self.EPSILON>=self.dyna_config['exploration']['min_epsilon']
						
						
						a = self.agent.choose_action(s, self.EPSILON)

						# take action
						s_, r, done, info = lobenv.step(a)
						ep_r = r+ep_r*discount

						# store current transition
						self.agent.store_transition(s, a, r, s_, done,initial)


						if done:
							self.episode+=1
							
							#agent should liquidate any remaining holdings and cancel orders - ncessary for correct balance calculation
							lobenv.liquidate()
							
							end_balance=lobenv.trader.balance
							profit=end_balance-start_balance
							
							# start update policy when memory has enough exps
							if self.agent.memory_counter > self.dyna_config['first_update']:
								self.agent.learn(EPSILON=self.EPSILON)
							
								#no planning before first update
								if self.planning and i_episode%self.dyna_config['model_update_freq']==0:
									if self.agent.memory_counter > self.dyna_config['batch_size']:
										
										if 'model' in self.dyna_config and self.dyna_config['model']=='tabular':
											#no model update stage when tabular
											pass
										else:
											self.agent.update_env_model()
											
									if i_episode%self.dyna_config['planning_freq']==0:
										for _ in range(self.planning_steps):
											if 'model' in self.dyna_config and self.dyna_config['model']=='tabular':
												self.agent.simulate_learn_tabular(EPSILON=self.EPSILON)
											else:
												self.agent.simulate_learn(EPSILON=self.EPSILON)
							#modify for fake cash
							#ep_r=ep_r+(1-lobenv.lamb)*fc
							
							#store,plot and display data
							self.store_train_data(i_episode,timestep,ep_r,profit)

							#plot results at visdom
							self.display_train_data(i_episode,timestep)
														
							
							#store checkpoint if necessary
							if i_episode>self.dyna_config['first_update']:
								self.checkpoint_make(i_episode)
							
							
							
							#check on stopping conditions
							if self.lr.profit==0 and self.lr.reward>5: 
								print('environment time:',lobenv.sess.time)
								raise ProfitWeird

							#check to revert
							if i_episode%250==0 and i_episode>1000:
								self.revert(i_episode)
							
							
							if self.stopping and i_episode>50: raise GetOutOfLoop
							break
							
						else:
							s = s_
							initial=False
						
			except GetOutOfLoop:
				print('stopping')
				
				pass
				
			except ProfitWeird:
				print('stopping, weird profit')

				pass
				
					
					

		
		def test_loop(self,train_episode,MaxEpisodes,start_episode=0,testm=False,lookback=20):
			
			agent=self.agent
			
			EPSILON=0
			total_steps = 0
			

			
			try:
				discount=self.dyna_config['discount']
			except TypeError:
				discount=self.agent.discount
			
			for i_episode in range(start_episode,MaxEpisodes):
				#no conesecutive test in same environment
				lobenv_test=self.env_selector(train_episode+self.test_counter,self.lobenvs)
				self.test_counter+=1
			
				try:
					self.agent.toggle_net(i_episode)
				except AttributeError:
					#not double q
					pass
				s,r0 = lobenv_test.reset()
				start_balance=lobenv_test.trader.balance
				ep_r = lobenv_test.lamb*r0
				#ep_r=0
				#if r0!=0: print('r0',r0)
				
				timestep = 0
				lob_start=lobenv_test.time
				self.info=[]
				
				while True:
					total_steps += 1

					a = agent.choose_action(s, EPSILON)

					# take action
					s_, r, done, info = lobenv_test.step(a)
					self.info.append(info)
					#agent.store_transition(s, a, r, s_, done,test=testm)
					ep_r = r+ep_r*discount
					
					timestep += 1

					if done:						
						end_time=lobenv_test.time
						lobenv_test.liquidate() #note the liquidation here. 
						end_balance=lobenv_test.trader.balance
						profit=end_balance-start_balance
						#self.rwd_test.append((lob_start,end_time,total_steps,i_episode,ep_r,profit,self.lobenv_test.initial_distance))
						self.lr=LossRecord(i_episode,timestep,ep_r,profit)
						self.rwd_dyna_test.append(self.lr)
						
						break
					else:
						s = s_
				

				
			self.stopping,self.median_test_loss,self.mean_test_loss=self.stopper(self.rwd_dyna_test,lookback=lookback,thresh=self.thresh)
			
			#plot test results
			self.plot_results_test(np.array([train_episode]),np.array([self.mean_test_loss]),np.array([self.median_test_loss]))
			
			
			print(f'TEST episode {train_episode}, median {self.median_test_loss} | mean | {self.mean_test_loss}')
				
		
		def plot_results_test(self,i_episode,mean_loss,median_loss):
			self.vis.line(X=i_episode, Y=mean_loss, win=self.test_loss_window, update='append',name='mean_test')
			self.vis.line(X=i_episode, Y=median_loss, win=self.test_loss_window, update='append',name='median_test')
			self.vis.get_window_data(self.train_loss_window)
				
				
		def checkpoint_make(self,i_episode):
			folder=self.folder
			#save every 1000 episodes 							
			if i_episode%1000==0 and i_episode>=1000:
				print(f'Saving checkpoint at episode {i_episode}')
				_=self.__checkpointModel(False,setup=True,tabular=True,memory=True,folder=folder)
				
			#check whether the reported mean_loss is better than best test_loss
			elif (self.mean_loss>max(0,self.best_rew[0]) or self.median_loss>max(0,self.best_rew[1])) and i_episode-self.best_rew[2]>5:
					
				
				#do a test
				if i_episode-self.last_test>self.dyna_config['model_update_freq']:
					self.test_loop(i_episode,self.lookback,lookback=self.lookback)
					self.last_test=i_episode
					
					
					#potentially save if a record breaker
					if max(self.mean_test_loss,self.median_test_loss)>max(0,self.best_rew[0])  and i_episode-self.best_rew[2]>10:
							
							#test agin - avoid winners curse - reuse some of the previous data
							self.test_loop(i_episode,2*self.lookback,lookback=3*self.lookback)
							if  self.mean_test_loss>max(0,self.best_rew[0]):
								print(f'Saving best checkpoint at episode {i_episode} with reward {self.mean_test_loss}')
								self.best_state_dict=self.__checkpointModel(True,setup=True,tabular=False,memory=True,folder=folder)
								self.best_rew=(self.mean_test_loss,self.median_test_loss,i_episode)
					
		def revert(self,episode):
		
				self.test_loop(episode,self.thresh,start_episode=0,testm=False)
				if self.mean_test_loss<0.8*self.best_rew[0]:
								#revert back to previous best
					print('REVERT BACK TO PRIOR NET')
					try:
						if self.agent.eval_net is not None:
							self.agent.learn_step_counter=self.best_state_dict['learn_step_counter']
							self.agent.eval_net.load_state_dict(self.best_state_dict['state_dict'])
							self.agent.target_net.load_state_dict(self.best_state_dict['state_dict'])
							self.agent.optimizer.load_state_dict(self.best_state_dict['optimizer'])
							
					except AttributeError:
							warnings.warn('no eval net for agent, skipping')
					try:
							self.agent.QNet0.load_state_dict(self.best_state_dict['Q0'])
							self.agent.QNet1.load_state_dict(self.best_state_dict['Q1'])
							self.agent.Q0optimizer.load_state_dict(self.best_state_dict['Q0Optim']),
							self.agent.Q1optimizer.load_state_dict(self.best_state_dict['Q1Optim']),
							self.agent.toggle_net(-1)
							
					except AttributeError:
							warnings.warn('not strict double q, skipping')
							


