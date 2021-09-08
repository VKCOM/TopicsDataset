import pickle

import matplotlib.pyplot as plt
from visualization.plots import plot_conf_int

passive_accs = []
passive_roc_aucs = []
passive_pr_aucs = []

passive_weight300_accs = []
passive_weight300_roc_aucs = []
passive_weight300_pr_aucs = []

ll_ideal_accs = []
ll_ideal_roc_aucs = []
ll_ideal_pr_aucs = []

ll_ideal_weight300_accs = []
ll_ideal_weight300_roc_aucs = []
ll_ideal_weight300_pr_aucs = []

for i in range(1, 2):
    state = pickle.load(open('statistic/worthiness/norm/passive_i2000_b20_q200_test' + str(i) + '.pkl', 'rb'))
    passive_accs.append([s['accuracy'] for s in state['performance_history']])
    passive_roc_aucs.append([s['roc_auc_score'] for s in state['performance_history']])
    passive_pr_aucs.append([s['pr_auc_score'] for s in state['performance_history']])

    state = pickle.load(open('statistic/worthiness/norm/passive_i2000_b20_q200_weight300_' + str(i) + '.pkl', 'rb'))
    passive_weight300_accs.append([s['accuracy'] for s in state['performance_history']])
    passive_weight300_roc_aucs.append([s['roc_auc_score'] for s in state['performance_history']])
    passive_weight300_pr_aucs.append([s['pr_auc_score'] for s in state['performance_history']])

    state = pickle.load(open('statistic/worthiness/norm/ll_ideal_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    ll_ideal_accs.append([s['accuracy'] for s in state['performance_history']])
    ll_ideal_roc_aucs.append([s['roc_auc_score'] for s in state['performance_history']])
    ll_ideal_pr_aucs.append([s['pr_auc_score'] for s in state['performance_history']])

    state = pickle.load(open('statistic/worthiness/norm/ll_ideal_i2000_b20_q200_weight300_' + str(i) + '.pkl', 'rb'))
    ll_ideal_weight300_accs.append([s['accuracy'] for s in state['performance_history']])
    ll_ideal_weight300_roc_aucs.append([s['roc_auc_score'] for s in state['performance_history']])
    ll_ideal_weight300_pr_aucs.append([s['pr_auc_score'] for s in state['performance_history']])

n_queries = state['n_queries']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_size_inches(15, 5, forward=True)
# plot_conf_int(passive_accs, 2000, 20, n_queries, 'passive', color='C0', ax=ax1)
# plot_conf_int(ll_ideal_accs, 2000, 20, n_queries, 'll_ideal', color='C1', ax=ax1)
plot_conf_int(passive_weight300_accs, 2000, 20, n_queries, 'passive (weight [1, 300])', color='C2', ax=ax1)
plot_conf_int(ll_ideal_weight300_accs, 2000, 20, n_queries, 'll_ideal (weight [1, 300])', color='C3', ax=ax1)

ax1.set_xlabel('labeled size')
ax1.set_ylabel('accuracy')
ax1.legend(loc='lower right')

# plot_conf_int(passive_roc_aucs, 2000, 20, n_queries, 'passive', color='C0', ax=ax2)
# plot_conf_int(ll_ideal_roc_aucs, 2000, 20, n_queries, 'll_ideal', color='C1', ax=ax2)
plot_conf_int(passive_weight300_roc_aucs, 2000, 20, n_queries, 'passive (weight [1, 300])', color='C2', ax=ax2)
plot_conf_int(ll_ideal_weight300_roc_aucs, 2000, 20, n_queries, 'll_ideal (weight [1, 300])', color='C3', ax=ax2)

ax2.set_xlabel('labeled size')
ax2.set_ylabel('roc auc')
ax2.legend(loc='lower right')

# plot_conf_int(passive_pr_aucs, 2000, 20, n_queries, 'passive', color='C0', ax=ax3)
# plot_conf_int(ll_ideal_pr_aucs, 2000, 20, n_queries, 'll_ideal', color='C1', ax=ax3)
plot_conf_int(passive_weight300_pr_aucs, 2000, 20, n_queries, 'passive (weight [1, 300])', color='C2', ax=ax3)
plot_conf_int(ll_ideal_weight300_pr_aucs, 2000, 20, n_queries, 'll_ideal (weight [1, 300]', color='C3', ax=ax3)

ax3.set_xlabel('labeled size')
ax3.set_ylabel('pr auc')
ax3.legend(loc='lower right')
fig.show()