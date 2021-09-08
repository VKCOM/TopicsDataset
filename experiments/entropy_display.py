import pickle
import matplotlib.pyplot as plt

from visualization.plots import plot_conf_int

least_confident_accs = []
entropy_accs = []
passive_accs = []
cluster_accs = []
margin_accs = []

for i in range(1, 6):
    state = pickle.load(open('statistic/least_confident_i500_b20_' + str(i) + '.pkl', 'rb'))
    least_confident_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/entropy_1_i500_b20_' + str(i) + '.pkl', 'rb'))
    entropy_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/passive_i500_b20_' + str(i) + '.pkl', 'rb'))
    passive_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/margin_i500_b20_' + str(i) + '.pkl', 'rb'))
    margin_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/cluster_i500_b20_' + str(i) + '.pkl', 'rb'))
    cluster_accs.append(state['performance_history'])

init_size = state['init_size']
n_queries = state['n_queries']
plot_conf_int(margin_accs, init_size, 20, n_queries, 'метод минимального отступа', color='C3')
plot_conf_int(least_confident_accs, init_size, 20, n_queries, 'метод наименьшей уверенности', color='C2')
plot_conf_int(entropy_accs, init_size, 20, n_queries, 'метод энтропии', color='C1')
plot_conf_int(passive_accs, init_size, 20, n_queries, 'без активного обучения', color='C0')
# plot_conf_int(cluster_accs, init_size, 20, n_queries, 'метод кластеризации', color='C4')

plt.xlabel('количество размеченных объектов')
plt.ylabel('точность модели')
plt.legend(loc='lower right')
plt.show()