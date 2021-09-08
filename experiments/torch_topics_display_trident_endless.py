import pickle
import matplotlib.pyplot as plt

from visualization.plots import plot_conf_int

passive_trident_based_accs = []
passive_512_accs = []
passive_e10_accs = []

bald_trident_x3_accs = []
bald_trident_e15_x3_accs = []
bald_trident_based_3_accs = []

bald_trident_bn_x3_accs = []
trident_passive_bn_accs = []

bald_trident_bn_es_accs = []
passive_trident_bn_es_accs = []

passive_trident_bn_e1_accs = []
bald_trident_x3_bn_e1_accs = []

margin_trident_based_bn_accs = []

margin_trident_sum_accs = []
margin_trident_sum_of_squares_accs = []
margin_trident_mult_accs = []

bald_trident_init_bn_x3_accs = []
bald_accs = []
margin_trident_sum_init_bn_accs = []
margin_trident_sum_of_squares_init_bn_accs = []
margin_trident_mult_init_bn_accs = []
margin_trident_mult_init_bn_accs_2 = []

margin_accs = []

min_n_queries = -1

for i in range(1, 6):

    # state = pickle.load(open('experiments/statistic/topics/torch/endless_queries/bald_trident_x3_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_trident_x3_accs.append(state['performance_history'])
    #
    # cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    #
    # if min_n_queries == -1 or cur_n_queries < min_n_queries:
    #     min_n_queries = cur_n_queries

    # state = pickle.load(open('statistic/topics/torch/endless_queries/bald_trident_based_3_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_trident_based_3_accs.append(state['performance_history'])
    #
    # cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    #
    # if min_n_queries == -1 or cur_n_queries < min_n_queries:
    #     min_n_queries = cur_n_queries
    #
    # state = pickle.load(open('experiments/statistic/topics/torch/endless_queries/trident_passive_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # passive_trident_based_accs.append(state['performance_history'])
    #
    # cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    #
    # if min_n_queries == -1 or cur_n_queries < min_n_queries:
    #     min_n_queries = cur_n_queries
    # #
    # state = pickle.load(open('statistic/topics/torch/endless_queries/bald_trident_x3_e15_i2000_b20_' + str(i) + '.pkl', 'rb'))
    # bald_trident_e15_x3_accs.append(state['performance_history'])
    #
    # cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    #
    # if min_n_queries == -1 or cur_n_queries < min_n_queries:
    #     min_n_queries = cur_n_queries
    #
    # state = pickle.load(open('statistic/topics/torch/endless_queries_bn/bald_trident_x3_e15_i2000_b20_' + str(i) + '.pkl', 'rb'))
    # bald_trident_bn_x3_accs.append(state['performance_history'])
    #
    # cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    #
    # if min_n_queries == -1 or cur_n_queries < min_n_queries:
    #     min_n_queries = cur_n_queries
    #
    # state = pickle.load(open('statistic/topics/torch/endless_queries_bn/trident_passive_e15_i2000_b20_' + str(i) + '.pkl', 'rb'))
    # trident_passive_bn_accs.append(state['performance_history'])
    #
    # cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    #
    # if min_n_queries == -1 or cur_n_queries < min_n_queries:
    #     min_n_queries = cur_n_queries

    # state = pickle.load(
    #     open('statistic/topics/torch/endless_queries_bn/trident_passive_es_i2000_b20_inter_' + str(i) + '.pkl', 'rb'))
    # passive_trident_bn_es_accs.append(state['performance_history'])
    #
    # cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    #
    # if min_n_queries == -1 or cur_n_queries < min_n_queries:
    #     min_n_queries = cur_n_queries
    #
    # state = pickle.load(
    #     open('statistic/topics/torch/endless_queries_bn/bald_trident_x3_es_i2000_b20_inter_' + str(i) + '.pkl', 'rb'))
    # bald_trident_bn_es_accs.append(state['performance_history'])
    #
    # cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    #
    # if min_n_queries == -1 or cur_n_queries < min_n_queries:
    #     min_n_queries = cur_n_queries
    #
    state = pickle.load(open('experiments/statistic/topics/torch/endless_queries_bn/bald_trident_x3_e1_i2000_b20_inter_' + str(i) + '.pkl', 'rb'))
    bald_trident_x3_bn_e1_accs.append(state['performance_history'])

    cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']

    if min_n_queries == -1 or cur_n_queries < min_n_queries:
        min_n_queries = cur_n_queries
    #

    state = pickle.load(open('experiments/statistic/topics/torch/endless_queries_bn/trident_passive_e1_i2000_b20_inter_' + str(i) + '.pkl', 'rb'))
    passive_trident_bn_e1_accs.append(state['performance_history'])

    cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    print('passive n queries:', cur_n_queries)

    if min_n_queries == -1 or cur_n_queries < min_n_queries:
        min_n_queries = cur_n_queries

    state = pickle.load(
        open('experiments/statistic/topics/torch/endless_queries_bn/margin_trident_based_e1_i2000_b20_inter_' + str(i) + '.pkl', 'rb'))
    margin_trident_based_bn_accs.append(state['performance_history'])

    cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']

    if min_n_queries == -1 or cur_n_queries < min_n_queries:
        min_n_queries = cur_n_queries

    # state = pickle.load(open('statistic/topics/torch_init_bn/margin_trident_sum_e1_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # margin_trident_sum_init_bn_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_init_bn/margin_trident_sum_of_squares_e1_i2000_b20_q200' + str(i) + '.pkl', 'rb'))
    # margin_trident_sum_of_squares_init_bn_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/endless_queries_bn/margin_trident_multiplication_e1_i2000_b20_2_' + str(i) + '.pkl', 'rb'))
    margin_trident_mult_accs.append(state['performance_history'])

    # state = pickle.load(
    #     open('statistic/topics/torch/endless_queries_bn/margin_trident_sum_of_squares_e1_i2000_b20_inter_' + str(i) + '.pkl', 'rb'))
    # margin_trident_sum_of_squares_accs.append(state['performance_history'])
    #
    # cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    # print('n queries', cur_n_queries)
    # if min_n_queries == -1 or cur_n_queries < min_n_queries:
    #     min_n_queries = cur_n_queries
    #
    # state = pickle.load(
    #     open('statistic/topics/torch/endless_queries_bn/margin_trident_multiplication_e1_i2000_b20_2_' + str(i) + '.pkl', 'rb'))
    # margin_trident_mult_accs.append(state['performance_history'])
    #
    # cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    #
    # if min_n_queries == -1 or cur_n_queries < min_n_queries:
    #     min_n_queries = cur_n_queries

    state = pickle.load(
        open('statistic/topics/torch/endless_queries_init_bn/margin_trident_multiplication_e1_i2000_b20_inter_' + str(
            i) + '.pkl', 'rb'))
    margin_trident_mult_init_bn_accs.append(state['performance_history'])

    cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    print('margin n queries:', cur_n_queries)
    if min_n_queries == -1 or cur_n_queries < min_n_queries:
        min_n_queries = cur_n_queries

    state = pickle.load(
        open('statistic/topics/torch/endless_queries_init_bn/margin_trident_multiplication_2_e1_i2000_b20_' + str(
            i) + '.pkl', 'rb'))
    margin_trident_mult_init_bn_accs_2.append(state['performance_history'])

    # cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    # print('margin n queries:', cur_n_queries)
    # if min_n_queries == -1 or cur_n_queries < min_n_queries:
    #     min_n_queries = cur_n_queries
    # #
    state = pickle.load(
        open('statistic/topics/torch/endless_queries_init_bn/bald_trident_x3_e1_i2000_b20_' + str(
            i) + '.pkl', 'rb'))
    bald_trident_init_bn_x3_accs.append(state['performance_history'])

    cur_n_queries = state['cur_n_queries'] if 'cur_n_queries' in state else state['n_queries']
    print('bald n queries:', cur_n_queries)
    if min_n_queries == -1 or cur_n_queries < min_n_queries:
        min_n_queries = cur_n_queries

    # state = pickle.load(open('statistic/topics/torch_init_bn/bald_x3_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # bald_accs.append(state['performance_history'])
    #
    state = pickle.load(open('statistic/topics/torch_init_bn/margin_e1_i2000_b20_inter_' + str(i) + '.pkl', 'rb'))
    margin_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/trident_passive_bs512_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # passive_512_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/trident_passive_e10_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # passive_e10_accs.append(state['performance_history'])

print('min n queries', min_n_queries)
# min_n_queries = 2000
# min_n_queries = 100
# min_n_queries = 500
# plot_conf_int(bald_trident_x3_accs, 2000, 20, 600, 'bald_trident', color='C4')
# plot_conf_int(bald_trident_based_3_accs, 2000, 20, min_n_queries, 'bald', color='C3')
# plot_conf_int(passive_trident_based_accs, 2000, 20, 400, 'passive', color='C0')
# plot_conf_int(bald_trident_e15_x3_accs, 2000, 20, min_n_queries, 'bald_trident e15', color='C5')

# plot_conf_int(bald_trident_e15_x3_accs, 2000, 20, min_n_queries, 'bald_trident bn e15', color='C6')
# plot_conf_int(trident_passive_bn_accs, 2000, 20, min_n_queries, 'passive bn e15', color='C7')

# plot_conf_int(passive_trident_bn_es_accs, 2000, 20, min_n_queries, 'passive trident bn es', color='C7')
# plot_conf_int(bald_trident_bn_es_accs, 2000, 20, min_n_queries, 'bald trident bn es', color='C8')

plot_conf_int(passive_trident_bn_e1_accs, 2000, 20, 200, 'passive', color='C0')
# plot_conf_int(bald_trident_x3_bn_e1_accs, 2000, 20, 2000, 'BALD', color='C1')
#
# plot_conf_int(margin_trident_based_bn_accs, 2000, 20, min_n_queries, 'margin bn', color='C2')

# plot_conf_int(margin_trident_sum_accs, 2000, 20, min_n_queries, 'batch normalization продолжает обучаться', color='C13')
# plot_conf_int(margin_trident_sum_of_squares_accs, 2000, 20, min_n_queries, 'margin trident sum of squares', color='C15')
# plot_conf_int(margin_trident_mult_accs, 2000, 20, 2000, 'batch normalization продолжает обучаться', color='C16')

# plot_conf_int(margin_trident_sum_init_bn_accs, 2000, 20, min_n_queries, 'batch normalization в режиме предсказания', color='C14')
# plot_conf_int(margin_trident_sum_of_squares_init_bn_accs, 2000, 20, min_n_queries, 'margin trident sum of squares', color='C15')
plot_conf_int(margin_trident_mult_init_bn_accs_2, 2000, 20, 200, 'margin с произведением', color='C16')
#
# plot_conf_int(bald_trident_init_bn_x3_accs, 2000, 20, 2000, 'BALD (с отключенной batch normalization)', color='C3')
# plot_conf_int(bald_accs, 2000, 20, min_n_queries, 'bald', color='C19')
plot_conf_int(margin_accs, 2000, 20, 200, 'margin', color='C2')
# plot_conf_int(passive_512_accs, 2000, 20, 200, '512', color='C7')
# plot_conf_int(passive_e10_accs, 2000, 20, 200, 'e10', color='C8')



plt.xlabel('размер размеченного набора данных')
plt.ylabel('валидационная точность')
plt.legend(loc='lower right')
plt.show()
