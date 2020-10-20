from utils.helpers import *
import matplotlib.pyplot as plt

def compute_metrics_and_plot(base_dir_path, num):
	structures = ["one_to_one", "one_to_many", "many_to_one"]
	labels = ["One to one", "Common cause", "Common effect"]
	mean_vectors = [[] for _ in range(3)]
	std_vectors = [[] for _ in range(3)]
	print(f"N {num}")

	plots_path = os.path.join(base_dir_path, "plots")
	for i in range(len(structures)):
		base_data_path = os.path.join(base_dir_path, structures[i], str(num), "mats")
		mat_files_paths = sorted([os.path.join(base_data_path, f) for f in os.listdir(base_data_path)])
		mat_files_paths = mat_files_paths[:1] + \
			mat_files_paths[2:] + [mat_files_paths[1]]
		total_hamming = []
		total_acc = []
		total_l2 = []
		global_xor_results = []
		global_acc_results = []
		global_l2_results = []
		for j in range(len(mat_files_paths)):
			data = read_dict_from_pickle(mat_files_paths[j])
			gt = data[f"gt_{j}"]
			beliefs = data[f"beliefs_{j}"]
			beliefs_boolean = transform_to_boolean_values(beliefs)
			gt_boolean = transform_to_boolean_values(gt)
			xor_results = metric_per_episode(gt=gt_boolean, beliefs=beliefs_boolean, operator="xor")
			acc_results = metric_per_episode(gt=gt_boolean, beliefs=beliefs_boolean, operator="equal")
			l2_results = l2_loss(gt, beliefs)
			total_hamming.append(xor_results[-1])
			total_acc.append(acc_results[-1])
			total_l2.append(l2_results[-1])
			global_xor_results.append(xor_results)
			global_acc_results.append(acc_results)
			global_l2_results.append(l2_results)
		mean_vectors[0].append(np.mean(global_xor_results, axis=0))
		mean_vectors[1].append(np.mean(global_acc_results, axis=0))
		mean_vectors[2].append(np.mean(global_l2_results, axis=0))
		std_vectors[0].append(np.std(global_xor_results, axis=0))
		std_vectors[1].append(np.std(global_acc_results, axis=0))
		std_vectors[2].append(np.std(global_l2_results, axis=0))
		print(f"{labels[i]}")
		print(f"Hamming {np.mean(total_hamming)} +- {np.std(total_hamming)}")
		print(f"Acc {np.mean(total_acc)} +- {np.std(total_acc)}")
		print(f"l2 loss {np.mean(total_l2)} +- {np.std(total_l2)}")
	metrics = ["hamming", "acc", "l2"]
	for i in range(len(mean_vectors)):
		x_axis = [k for k in range(len(mean_vectors[i][0]))]
		filename = os.path.join(plots_path, f"{metrics[i]}_{num}")
		plot_measures(x_axis=x_axis, mean_vecs=mean_vectors[i], std_dev_vectors=std_vectors[i], labels=labels, filename=filename)

def compute_metrics_and_plot_disease_treatment(paths):
	comparison_mean = []
	comparison_std = []
	plots_path = os.path.join("results", "plots")
	for base_dir_path in paths:
		base_data_path = os.path.join(base_dir_path, "mats")
		mat_files_paths = sorted([os.path.join(base_data_path, f)
							for f in os.listdir(base_data_path)])
		global_xor_results = []
		global_acc_results = []
		global_l2_results = []
		global_rewards = []
		mean_vectors = [_ for _ in range(4)]
		std_vectors = [_ for _ in range(4)]
		for j in range(len(mat_files_paths)):
			data = read_dict_from_pickle(mat_files_paths[j])
			gt = data[f"gt_{j}"]
			beliefs = data[f"beliefs_{j}"]
			beliefs_boolean = transform_to_boolean_values(beliefs)
			gt_boolean = transform_to_boolean_values(gt)
			xor_results = metric_per_episode(gt=gt_boolean, beliefs=beliefs_boolean, operator="xor")
			acc_results = metric_per_episode(gt=gt_boolean, beliefs=beliefs_boolean, operator="equal")
			l2_results = l2_loss(gt, beliefs)
			global_xor_results.append(xor_results)
			global_acc_results.append(acc_results)
			global_l2_results.append(l2_results)
			rewards = data[f"rewards_{j}"]
			if len(rewards) == 50:
				rewards = [0] + rewards
			global_rewards.append(rewards)
		mean_vectors[0] = np.mean(global_xor_results, axis=0)
		mean_vectors[1] = np.mean(global_acc_results, axis=0)
		mean_vectors[2] = np.mean(global_l2_results, axis=0)
		mean_vectors[3] = np.mean(global_rewards, axis=0)
		std_vectors[0] = np.std(global_xor_results, axis=0)
		std_vectors[1] = np.std(global_acc_results, axis=0)
		std_vectors[2] = np.std(global_l2_results, axis=0)
		std_vectors[3] = np.std(global_rewards, axis=0)
		comparison_mean.append(mean_vectors)
		comparison_std.append(std_vectors)
	metrics = ["hamming", "acc", "l2", "rewards"]
	labels = ["Random", "Exponential decay", "Linear decay"]
	for i in range(len(comparison_mean[0])):
		x_axis = [k for k in range(len(mean_vectors[i]))]
		filename = os.path.join(plots_path, f"{metrics[i]}")
		print(x_axis, mean_vectors[i])
		plot_measures(x_axis=x_axis, mean_vecs=[comparison_mean[0][i], comparison_mean[1][i], comparison_mean[2][i]], std_dev_vectors=[
		              comparison_std[0][i], comparison_std[1][i], comparison_std[2][i]], labels=labels, filename=filename, legend=True)


def compare_rewards(dirs, labels, filename="reward_comparison"):
	"""
	docstring
	"""
	mean_vectors = []
	std_vectors = []
	plots_path = os.path.join("results", "plots", filename)
	for base_dir_path in dirs:
		base_data_path = os.path.join(base_dir_path, "mats")
		mat_files_paths = sorted([os.path.join(base_data_path, f)
							for f in os.listdir(base_data_path)])
		global_rewards = []
		for j in range(len(mat_files_paths)):
			data = read_dict_from_pickle(mat_files_paths[j])
			rewards = data[f"rewards_{j}"]
			if len(rewards) == 50: 
				rewards = [0] + rewards
				len(global_rewards)
			global_rewards.append(rewards)
		mean_vectors.append(np.mean(global_rewards, axis=0))
		std_vectors.append(np.std(global_rewards, axis=0))
	x_axis = [k for k in range(len(mean_vectors[0]))]
	print(len(mean_vectors[0]))
	print(len(mean_vectors[1]))
	print(len(mean_vectors[2]))
	plot_measures(x_axis=x_axis, mean_vecs=mean_vectors, std_dev_vectors=std_vectors, labels=labels, filename=plots_path, legend=True)

def create_heatmaps(path, num, n=50, bin_size=5):
	"""
	Crea heatmaps para el GT y las creencias.
	"""
	data = read_dict_from_pickle(path)
	row = []
	for t in range(0, n + 1, n // bin_size):
		gt, beliefs_t = beliefs_to_mat(data, num, timestep=t)
		row.append(beliefs_t)
	row.append(gt)
	return row
	# 	plot_heatmap(beliefs_t, os.path.join(outpath, f"beliefs_heatmap_t_{t}_{num}"))
	# plot_heatmap(gt, os.path.join(outpath, f"gt_heatmap_{num}"))


def create_grid_of_heatmaps(base_dir_path, structure, nums=[5, 7, 9], bin_size=5):
	"""
	docstring
	"""
	plot_path = os.path.join(base_dir_path, "plots", structure)
	fig, axs = plt.subplots(len(nums), bin_size + 2)
	print(structure)
	for i in range(len(nums)):
		print(f"N {nums[i]}")
		base_data_path = os.path.join(base_dir_path, structure, str(nums[i]), "mats")
		mat_files_paths =[os.path.join(base_data_path, f) for f in os.listdir(base_data_path)]
		plot_row = create_heatmaps(mat_files_paths[-1], nums[i])
		for j in range(len(plot_row)):
			axs[i, j].set_ylabel('')
			axs[i, j].set_xlabel('')
			add_heatmap(plot_row[j], axs[i, j])
	plt.savefig("{}.png".format(plot_path), bbox_inches='tight')
	plt.close()
if __name__ == "__main__":
    # for i in [5]:#, 7, 9]:
    #     compute_metrics_and_plot("results/light-switches-best-action/", i)
	# compute_metrics_and_plot_disease_treatment(
	# 	"results/disease-treatment-random-action", "results/disease-treatment-best-action")
	# dirs = ["results/disease-treatment-linear",
    #      "results/disease-treatment-exponential", "results/qlearning-disease-linear", "results/qlearning-disease-exponential"]
	# labels = ["Linear", "Exponential", "Q-learning linear", "Q-learning exponential"]
	# compare_rewards(dirs, labels)
	# dirs = ["results/disease-treatment-random-action", "results/disease-treatment-exponential",
    #      "results/disease-treatment-linear"]
	# compute_metrics_and_plot_disease_treatment(dirs)
	structures = ["one_to_one", "one_to_many", "many_to_one"]
	for structure in structures:
		create_grid_of_heatmaps(base_dir_path="results/light-switches", structure=structure)
