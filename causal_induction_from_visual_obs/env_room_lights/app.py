# -*- coding: utf-8 -*-
import argparse

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from q_learning import QLearning, MOD_EPISODE
from q_learning_causal import QlearningAssisted, QlearningWrongAssited, QlearningPartiallyAssited

np.random.seed(42)

def main():
	parser = argparse.ArgumentParser(description='Run Q-learning and Q-learning CM light switch problem')
	parser.add_argument("--stochastic", help="change to simple stochastic enviroments (0.8 prob of do the choosen action)",\
						action="store_true")
	parser.add_argument("--episodes", type=int, default=1000, help="# of episodes per experiment")
	parser.add_argument("--experiments", type=int, default=5, help="# of experiments")
	parser.add_argument("--num", type=int, default=5, help="Number of switches")
	parser.add_argument("--structure", type=str, default="one_to_one", help="structure of the graph")
	parser.add_argument("--stat_test", action="store_true", help="Compute the Mann-Whitney rank test to check if some algorithm reach the goal faster. Need > 20 experiments")
	parser.add_argument("-v", "--verbose", action="store_true", help="Verbosite activated")
	args = parser.parse_args()
	episodes = args.episodes
	num = args.num 
	structure = args.structure
	number_of_experiments = args.experiments
	total_rewards = [[] for i in range(4)]
	# time_to_reach = [[] for i in range(4)]
	for i in range(number_of_experiments):
		if args.verbose:
			print("Running experiment {}/{}".format(i +  1, number_of_experiments))
		# print("QLearning")
		rewards_q_learning, episode_t_q = QLearning(episodes=episodes, num=num, structure=structure, filename="Qlearning").train()
		# print("QlearningAssisted")
		rewards_q_learning_causal, episode_t_q_causal = QlearningAssisted(episodes=episodes, num=num, structure=structure, filename="QlearningAssisted").train()
		rewards_q_learning_causal_partial, episode_t_q_causal_partial = QlearningPartiallyAssited(episodes=episodes, num=num, structure=structure, filename="QlearningAssisted").train()
		rewards_q_learning_causal_wrong, episode_t_q_causal_wrong = QlearningWrongAssited(episodes=episodes, num=num, structure=structure, filename="QlearningAssisted").train()
		total_rewards[0].append(rewards_q_learning)
		total_rewards[1].append(rewards_q_learning_causal)
		total_rewards[2].append(rewards_q_learning_causal_partial)
		total_rewards[3].append(rewards_q_learning_causal_wrong)
		# time_to_reach[0].append(episode_t_q)
		# time_to_reach[1].append(episode_t_q_causal)
		# total_rewards[2].append(QLearning(episodes=episodes).train(use_reward_feedback=True))

	scale_x = len(np.mean(total_rewards[0], axis=0))
	plot_x_axis = MOD_EPISODE * (np.arange(scale_x))
	
	mean_vanilla_q_leaning = np.mean(total_rewards[0], axis=0) 
	std_vanilla_q_leaning = np.std(total_rewards[0], axis=0) 
	
	mean_causal_q_leaning = np.mean(total_rewards[1], axis=0) 
	std_causal_q_leaning = np.std(total_rewards[1], axis=0) 
	

	mean_causal_q_leaning_partial = np.mean(total_rewards[2], axis=0) 
	std_causal_q_leaning_partial = np.std(total_rewards[2], axis=0) 
	

	mean_causal_q_leaning_wrong = np.mean(total_rewards[3], axis=0) 
	std_causal_q_leaning_wrong = np.std(total_rewards[3], axis=0) 
	

	plt.plot(plot_x_axis, mean_vanilla_q_leaning, label="Vanilla Q-learning", color="#CC4F1B")
	plt.fill_between(plot_x_axis, mean_vanilla_q_leaning - std_vanilla_q_leaning,\
					mean_vanilla_q_leaning + std_vanilla_q_leaning,\
                    alpha=0.2, edgecolor='#CC4F1B', facecolor='#FF9848')

	plt.plot(plot_x_axis, mean_causal_q_leaning, label="Assisted Q-learning", color="#2980b9")
	plt.fill_between(plot_x_axis, mean_causal_q_leaning - std_causal_q_leaning,\
					mean_causal_q_leaning + std_causal_q_leaning,\
                    alpha=0.2, edgecolor='#2980b9', facecolor='#3498db')
	
	plt.plot(plot_x_axis, mean_causal_q_leaning_partial, label="Partially Assisted Q-learning", color="#138d75")
	plt.fill_between(plot_x_axis, mean_causal_q_leaning_partial - std_causal_q_leaning_partial,\
					mean_causal_q_leaning_partial + std_causal_q_leaning_partial,\
                    alpha=0.2, edgecolor='#138d75', facecolor='#27ae60')

	plt.plot(plot_x_axis, mean_causal_q_leaning_wrong, label="Wrong Assisted Q-learning", color="#f39c12")
	plt.fill_between(plot_x_axis, mean_causal_q_leaning_wrong - std_causal_q_leaning_wrong,\
					mean_causal_q_leaning_wrong + std_causal_q_leaning_wrong,\
                    alpha=0.2, edgecolor='#f39c12', facecolor='#f1c40f')



	# plt.plot(plot_x_axis, np.mean(total_rewards[2], axis=0), color="red", label="Q-learning + CM + Feedback Revisited")
	plt.xlabel('Episodes')
	plt.ylabel('Average Reward')
	plt.legend()
	plt.title('Average Reward Comparison {} {}'.format("Stochastic" if args.stochastic else "Deterministic", structure))
	plt.savefig("plots/comparison{}_{}_{}.jpg".format("Stochastic" if args.stochastic else "Deterministic", structure, num))     
	plt.close()
	# if args.stat_test and number_of_experiments > 20:
	# 	print("Computing Mann-Whitney rank test.")
	# 	cm_times = time_to_reach[1]
	# 	vanilla_times = time_to_reach[0]
	# 	for i in range(number_of_experiments):
	# 		if cm_times[i] == None:
	# 			cm_times[i] = episodes
	# 		if vanilla_times[i] == None:
	# 			vanilla_times[i] = episodes
	# 	print("Vanilla average episode to reach goal reward: {}".format(np.mean(vanilla_times)))
	# 	print("Vanilla standard deviation episode to reach goal reward: {}".format(np.std(vanilla_times)))
	# 	print("CM average episode to reach goal reward: {}".format(np.mean(cm_times)))
	# 	print("CM standard deviation episode to reach goal reward: {}".format(np.std(cm_times)))
	# 	print(stats.mannwhitneyu(cm_times, vanilla_times))
	# 	plt.plot(np.arange(number_of_experiments), vanilla_times, label="Vanilla Q-learning")
	# 	plt.plot(np.arange(number_of_experiments), cm_times, label="Q-learning + CM")
	# 	plt.xlabel('Experiments')
	# 	plt.ylabel('Time to reach the goal (episode)')
	# 	plt.legend()
	# 	plt.title('Goal Reached'.format("Stochastic" if args.stochastic else "Deterministic"))
	# 	plt.savefig("goal_reward_comparison{}.jpg".format("Stochastic" if args.stochastic else "Deterministic"))     
	# 	plt.close()

if __name__ == '__main__':
	main()

