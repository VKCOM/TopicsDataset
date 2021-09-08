import pickle
import matplotlib.pyplot as plt

from visualization.plots import plot_conf_int

bald_trident_accs = []
bald_trident_mc_accs = []

bald_trident_based_accs = []
bald_trident_based_5_accs = []
bald_trident_based_10_accs = []

bald_trident_x3_accs = []
bald_trident_x3_e10_accs = []

bald_trident_x5_accs = []
trident_passive_accs = []

margin_e1_accs = []
margin_e5_accs = []
margin_e10_accs = []

for i in range(1, 6):

    state = pickle.load(open('statistic/topics/torch/d128/bald_trident_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_trident_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_trident_mc_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_trident_mc_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_trident_based_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_trident_based_accs.append(state['performance_history'])

    state = pickle.load(
        open('statistic/topics/torch/d128/bald_trident_based_5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_trident_based_5_accs.append(state['performance_history'])

    state = pickle.load(
        open('statistic/topics/torch/d128/bald_trident_based_10_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_trident_based_10_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_trident_x3_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_trident_x3_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_trident_x3_e10_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_trident_x3_e10_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_trident_x5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_trident_x5_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/trident_passive_i2000_b20_q100_' + str(i) + '.pkl', 'rb'))
    trident_passive_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/margin_trident_based_e1_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    margin_e1_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/margin_trident_based_e5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    margin_e5_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/margin_trident_based_e10_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    margin_e10_accs.append(state['performance_history'])

n_instances = state['n_instances']
n_queries = state['n_queries']

plot_conf_int(trident_passive_accs, 2000, 20, n_queries, 'passive', color='C0')

# plot_conf_int(bald_trident_accs, 2000, 20, n_queries, 'bald_trident', color='C1')
# plot_conf_int(bald_trident_mc_accs, 2000, 20, n_queries, 'bald_trident_mc', color='C2')

# plot_conf_int(bald_trident_x3_accs, 2000, 20, n_queries, 'bald_trident_x3', color='C4')

# plot_conf_int(bald_trident_x5_accs, 2000, 20, n_queries, 'bald_trident_x5', color='C4')

plot_conf_int(bald_trident_based_accs, 2000, 20, n_queries, 'bald', color='C3')
# plot_conf_int(bald_trident_based_5_accs, 2000, 20, n_queries, 'bald_trident_based_5', color='C6')

plot_conf_int(margin_e1_accs, 2000, 20, n_queries, 'margin', color='C2')
# plot_conf_int(margin_e5_accs, 2000, 20, n_queries, 'margin_e5', color='C8')
# plot_conf_int(margin_e10_accs, 2000, 20, n_queries, 'margin_e10', color='C9')

# plot_conf_int(bald_trident_x3_e10_accs, 2000, 20, n_queries, 'bald_trident_x3_e10', color='C10')


plt.xlabel('labeled size')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()
