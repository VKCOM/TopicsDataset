import pickle
import matplotlib.pyplot as plt

from visualization.plots import plot_conf_int

passive_accs = []

bald_trident_accs = []
bald_accs = []

margin_accs = []
margin_mult_accs = []
margin_mixed_accs = []

margin_non_mtl_accs = []

bald_trident_bn_accs = []

min_n_queries = -1

for i in range(1, 6):
    state = pickle.load(open('experiments/statistic/topics/torch/endless_queries_bn/trident_passive_e1_i2000_b20_inter_' + str(i) + '.pkl', 'rb'))
    passive_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch_mtl_init_bn/bald_x3_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    bald_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch_mtl_init_bn/bald_trident_x3_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    bald_trident_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch_mtl_init_bn/margin_i2000_b20_q400_' + str(i) + '.pkl', 'rb'))
    margin_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch_mtl_init_bn/margin_trident_multiplication_i2000_b20_q400_' + str(i) + '.pkl', 'rb'))
    margin_mult_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch_mtl_init_bn/margin_mixed_i2000_b20_q400_' + str(i) + '.pkl', 'rb'))
    margin_mixed_accs.append(state['performance_history'])

    state = pickle.load(open('experiments/statistic/topics/torch_bn/margin_inter_i2000_b20_q100_' + str(i) + '.pkl', 'rb'))
    margin_non_mtl_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch_mtl_init_bn/bald_trident_bn_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    bald_trident_bn_accs.append(state['performance_history'])


print('min n queries', min_n_queries)
min_n_queries = 200

plot_conf_int(passive_accs, 2000, 20, min_n_queries, 'пассивное обучение', color='C0')
print(passive_accs[0][:10])
plot_conf_int(bald_trident_accs, 2000, 20, min_n_queries, 'BALD с использованием вспомогательных выходов', color='C4')
plot_conf_int(bald_accs, 2000, 20, min_n_queries, 'BALD', color='C3')

# plot_conf_int(margin_mult_accs, 2000, 20, min_n_queries, 'margin с произведением', color='C6')
# plot_conf_int(margin_accs, 2000, 20, min_n_queries, 'margin', color='C2')
# plot_conf_int(margin_mixed_accs, 2000, 20, 400, 'margin mixed', color='C4')
# plot_conf_int(margin_non_mtl_accs, 2000, 20, 400, 'margin single head', color='C5')
#
# plot_conf_int(bald_trident_bn_accs, 2000, 20, 200, 'bald_trident_bn', color='C6')


plt.xlabel('размер размеченного набора данных')
plt.ylabel('валидационная точность')
plt.legend(loc='lower right')
plt.show()
