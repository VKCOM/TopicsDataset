import pickle

import matplotlib.pyplot as plt
from visualization.plots import plot_conf_int

passive_weight300_accs = []
passive_weight300_roc_aucs = []
passive_weight300_pr_aucs = []


margin_weight300_accs = []
margin_weight300_roc_aucs = []
margin_weight300_pr_aucs = []

for i in range(1, 6):
    state = pickle.load(open('statistic/topics_worthiness/trident/passive_i2000_b20_q200_weight300_' + str(i) + '.pkl', 'rb'))
    passive_weight300_accs.append([s['accuracy_topic'] for s in state['performance_history']])
    passive_weight300_roc_aucs.append([s['roc_auc_worthiness'] for s in state['performance_history']])
    passive_weight300_pr_aucs.append([s['pr_auc_worthiness'] for s in state['performance_history']])

    state = pickle.load(open('statistic/topics_worthiness/trident/margin_i2000_b20_q200_weight300_' + str(i) + '.pkl', 'rb'))
    margin_weight300_accs.append([s['accuracy_topic'] for s in state['performance_history']])
    margin_weight300_roc_aucs.append([s['roc_auc_worthiness'] for s in state['performance_history']])
    margin_weight300_pr_aucs.append([s['pr_auc_worthiness'] for s in state['performance_history']])


n_queries = state['n_queries']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_size_inches(15, 5, forward=True)
plot_conf_int(passive_weight300_accs, 2000, 20, n_queries, 'passive', color='C0', ax=ax1)
plot_conf_int(margin_weight300_accs, 2000, 20, n_queries, 'margin', color='C2', ax=ax1)

ax1.set_xlabel('labeled size')
ax1.set_ylabel('topic accuracy')
ax1.legend(loc='lower right')

# plot_conf_int(passive_roc_aucs, 2000, 20, n_queries, 'passive', color='C0', ax=ax2)
# plot_conf_int(ll_ideal_roc_aucs, 2000, 20, n_queries, 'll_ideal', color='C1', ax=ax2)
plot_conf_int(passive_weight300_roc_aucs, 2000, 20, n_queries, 'passive', color='C0', ax=ax2)
plot_conf_int(margin_weight300_roc_aucs, 2000, 20, n_queries, 'margin', color='C2', ax=ax2)

ax2.set_xlabel('labeled size')
ax2.set_ylabel('roc auc worthiness')
ax2.legend(loc='lower right')

# plot_conf_int(passive_pr_aucs, 2000, 20, n_queries, 'passive', color='C0', ax=ax3)
# plot_conf_int(ll_ideal_pr_aucs, 2000, 20, n_queries, 'll_ideal', color='C1', ax=ax3)
plot_conf_int(passive_weight300_pr_aucs, 2000, 20, n_queries, 'passive', color='C0', ax=ax3)
plot_conf_int(margin_weight300_pr_aucs, 2000, 20, n_queries, 'margin', color='C2', ax=ax3)

ax3.set_xlabel('labeled size')
ax3.set_ylabel('pr auc worthiness')
ax3.legend(loc='lower right')
fig.show()