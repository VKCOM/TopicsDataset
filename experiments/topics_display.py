import pickle

import matplotlib.pyplot as plt
from visualization.plots import plot_conf_int

passive_accs = []
lc_accs = []
margin_accs = []
entropy_accs = []
entropy_top_accs = {3: [], 4: [], 5: []}

for i in range(1, 6):
    state = pickle.load(open('statistic/topics/keras/passive_i2000_b20_' + str(i) + '.pkl', 'rb'))
    passive_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/keras/least_confident_i2000_b20_' + str(i) + '.pkl', 'rb'))
    lc_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/keras/margin_i2000_b20_' + str(i) + '.pkl', 'rb'))
    margin_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/keras/entropy_i2000_b20_' + str(i) + '.pkl', 'rb'))
    entropy_accs.append(state['performance_history'])

for j in range(1, 6):
    for i in range(3, 6):
        state = pickle.load(open('statistic/topics/keras/entropy_top_' + str(i) + '_i2000_b20_' + str(j) + '.pkl', 'rb'))
        entropy_top_accs[i].append(state['performance_history'])

n_instances = state['n_instances']
n_queries = state['n_queries']

plot_conf_int(passive_accs, 2000, 20, n_queries, 'passive', color='C0')
# plot_conf_int(lc_accs, 2000, 20, n_queries, 'least_confident', color='C1')
# plot_conf_int(margin_accs, 2000, 20, n_queries, 'margin', color='C2')
# plot_conf_int(entropy_top_accs[3], 2000, 20, n_queries, 'entropy_top_3', color='C3')
# plot_conf_int(entropy_top_accs[4], 2000, 20, n_queries, 'entropy_top_4', color='C4')
# plot_conf_int(entropy_top_accs[5], 2000, 20, n_queries, 'entropy_top_5', color='C5')

# plot_conf_int(entropy_accs, 2000, 20, n_queries, 'entropy')

plt.xlabel('labeled size')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()
