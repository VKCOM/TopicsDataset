import pickle

import matplotlib.pyplot as plt
from visualization.plots import plot_conf_int

passive_topic_accs = []
passive_worthiness_roc_aucs = []

for i in range(1, 2):
    state = pickle.load(open('statistic/topics_worthiness/trident/passive_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))

    print('performance history:', state['performance_history'])

    passive_topic_accs.append([s['accuracy_topic'] for s in state['performance_history']])
    passive_worthiness_roc_aucs.append([s['roc_auc_worthiness'] for s in state['performance_history']])

n_queries = state['n_queries']

plot_conf_int(passive_topic_accs, 2000, 20, n_queries, 'passive', color='C0')

plt.xlabel('labeled size')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()

plot_conf_int(passive_worthiness_roc_aucs, 2000, 20, n_queries, 'passive', color='C0')

plt.xlabel('labeled size')
plt.ylabel('roc auc')
plt.legend(loc='lower right')
plt.show()

