import pickle
import matplotlib.pyplot as plt

from visualization.plots import plot_conf_int

entropy_accs = []
ranked_accs = []

for i in range(1, 6):
    state = pickle.load(open('statistic/entropy_batch_' + str(i) + '.pkl', 'rb'))
    entropy_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/ranked_batch_' + str(i) + '.pkl', 'rb'))
    ranked_accs.append(state['performance_history'])

init_size = state['init_size']
n_queries = state['n_queries']
plot_conf_int(entropy_accs, init_size, 10, n_queries, 'entropy')
plot_conf_int(ranked_accs, init_size, 10, n_queries, 'ranked uns')
plt.xlabel('labeled size')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()
