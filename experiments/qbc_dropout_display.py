import pickle
import matplotlib.pyplot as plt

from visualization.plots import plot_conf_int

entropy_accs = []
qbc_dropout_accs = []

for i in range(1, 5):
    state = pickle.load(open('statistic/entropy_test_state_' + str(i) + '.pkl', 'rb'))
    entropy_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/qbc_test_state_' + str(i) + '.pkl', 'rb'))
    qbc_dropout_accs.append(state['performance_history'])

init_size = state['init_size']
n_instances = state['n_instances']
n_queries = state['n_queries']

plot_conf_int(entropy_accs, init_size, 10, n_queries, 'entropy')
plot_conf_int(qbc_dropout_accs, init_size, 10, n_queries, 'qbc dropout')
plt.xlabel('labeled size')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()
