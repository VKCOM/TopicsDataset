import pickle
import matplotlib.pyplot as plt

from visualization.plots import plot_conf_int

entropy_accs = []
passive_accs = []
cluster_accs = []
passive_bs_accs = []
entropy_bs_accs = []
cluster_bs_accs = []

for i in range(1, 6):
    state = pickle.load(open('statistic/passive_bs_i500_b20_' + str(i) + '.pkl', 'rb'))
    passive_bs_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/passive_i500_b20_' + str(i) + '.pkl', 'rb'))
    passive_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/entropy_i500_b20_' + str(i) + '.pkl', 'rb'))
    entropy_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/entropy_bs_i500_b20_' + str(i) + '.pkl', 'rb'))
    entropy_bs_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/cluster_i500_b20_' + str(i) + '.pkl', 'rb'))
    cluster_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/cluster_bs_i500_b20_' + str(i) + '.pkl', 'rb'))
    cluster_bs_accs.append(state['performance_history'])

init_size = state['init_size']
n_instances = state['n_instances']
n_queries = state['n_queries']
print(n_instances)
plot_conf_int(passive_accs, init_size, 20, n_queries, 'passive')
plot_conf_int(passive_bs_accs, init_size, 20, n_queries, 'passive_bs')
plot_conf_int(entropy_accs, init_size, 20, n_queries, 'entropy')
plot_conf_int(entropy_bs_accs, init_size, 20, n_queries, 'entropy_bs')
plot_conf_int(cluster_accs, init_size, 20, n_queries, 'cluster')
plot_conf_int(cluster_bs_accs, init_size, 20, n_queries, 'cluster_bs')


plt.xlabel('labeled size')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()
