import pickle

import matplotlib.pyplot as plt
from visualization.plots import plot_conf_int

passive_accs = []
passive_sf_accs = []
passive_up_accs = []
passive_e5_accs = []
passive_e10_accs = []

passive_weighted_accs = []
passive_shuffled_accs = []
passive_bs_accs = []
lc_accs = []

margin_accs = []
margin_sf_accs = []
margin_up_accs = []

margin_weighted_accs = []
entropy_accs = []
entropy_sf_accs = []

qbc_lc5_accs = []
qbc_margin5_accs = []
qbc_entropy5_accs = []

bald_lc5_accs = []
bald_entropy3_accs = []
bald_entropy5_accs = []
bald_margin5_accs = []

bald_margin10_accs = []
bald_entropy10_accs = []
bald_modal_accs = []
bald_modal_non_mc_accs = []

bald_trident_accs = []
bald_trident_mc_accs = []

bald_trident_based_accs = []
bald_trident_based_5_accs = []
bald_trident_based_10_accs = []

bald_trident_x3_accs = []
bald_trident_x5_accs = []
trident_passive_accs = []

diversity_img_accs = []
diversity_txt_accs = []
diversity_concat_accs = []

sud1000_accs = []
sud1000_ad64_entropy_accs = []
sud1000_ad64_margin_accs = []


for i in range(1, 6):
    state = pickle.load(open('statistic/topics/torch/d128/passive_i2000_b20_q100_' + str(i) + '.pkl', 'rb'))
    passive_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/passive_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    passive_sf_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch/d128/entropy_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # entropy_sf_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/passive_i2000_b20_q100_sf512_e10_' + str(i) + '.pkl', 'rb'))
    passive_e10_accs.append(state['performance_history'])


    # state = pickle.load(open('statistic/topics/torch/d128/diversity_img_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # diversity_img_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/diversity_txt_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # diversity_txt_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/diversity_concat_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # diversity_concat_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch/d128/bald_lc5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_lc5_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/bald_margin5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_margin5_accs.append(state['performance_history'])
    #

    state = pickle.load(open('statistic/topics/torch/d128/bald_entropy3_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_entropy3_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch/d128/bald_entropy5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_entropy5_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/sud1000_trivial_encode_entropy_i2000_b20_q100_' + str(i) + '.pkl', 'rb'))
    # sud1000_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/sud1000_trivial_encode_ad64_entropy_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # sud1000_ad64_entropy_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/sud1000_trivial_encode_ad64_margin_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # sud1000_ad64_margin_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch/d128/bald_modal_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_modal_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/bald_modal_non_mc_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_modal_non_mc_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/bald_trident_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_trident_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_trident_mc_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_trident_mc_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_trident_based_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_trident_based_accs.append(state['performance_history'])

    # state = pickle.load(
    #     open('statistic/topics/torch/d128/bald_trident_based_5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_trident_based_5_accs.append(state['performance_history'])
    #
    # state = pickle.load(
    #     open('statistic/topics/torch/d128/bald_trident_based_10_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_trident_based_10_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_trident_x3_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_trident_x3_accs.append(state['performance_history'])

    # state = pickle.load(
    #     open('statistic/topics/torch/d128/bald_trident_x5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_trident_x5_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/trident_passive_i2000_b20_q100_' + str(i) + '.pkl', 'rb'))
    trident_passive_accs.append(state['performance_history'])

    # #
    # state = pickle.load(open('statistic/topics/torch/d128/least_confident_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # lc_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/qbc_lc5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # qbc_lc5_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/bald_lc5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_lc5_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/qbc_margin5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # qbc_margin5_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/entropy_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # entropy_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch/d128/qbc_entropy5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # qbc_entropy5_accs.append(state['performance_history'])
    #
    # # state = pickle.load(open('statistic/topics/torch/d128/bald_margin10_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_margin10_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/bald_entropy10_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_entropy10_accs.append(state['performance_history'])
    #

    # state = pickle.load(open('statistic/topics/torch/d128/passive_i2000_b20_q100_shuffled_fit_' + str(i) + '.pkl', 'rb'))
    # passive_shuffled_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch/d128/passive_bs_i2000_b20_' + str(i) + '.pkl', 'rb'))
    # passive_bs_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch/d128/least_confident_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # lc_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch/d128/margin_i2000_b20_q100_' + str(i) + '.pkl', 'rb'))
    # margin_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/margin_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    margin_sf_accs.append(state['performance_history'])


n_instances = state['n_instances']
n_queries = state['n_queries']

plot_conf_int(passive_accs, 2000, 20, n_queries, 'пассивное обучение', color='C0')
# plot_conf_int(passive_sf_accs, 2000, 20, n_queries, 'passive', color='C0')
# plot_conf_int(passive_sf_accs, 2000, 20, n_queries, 'passive', color='C0')
# plot_conf_int(passive_sf_accs, 2000, 20, n_queries, 'passive', color='C0')

# plot_conf_int(entropy_sf_accs, 2000, 20, n_queries, 'entropy', color='C1')
# plot_conf_int(trident_passive_accs, 2000, 20, n_queries, 'passive_trident', color='C0')
# plot_conf_int(passive_e5_accs, 2000, 20, n_queries, 'passive sf 5', color='C2')
plot_conf_int(passive_e10_accs, 2000, 20, n_queries, 'пассивное обучение + изменяемый batch size', color='C3')

# plot_conf_int(passive_up_accs, 2000, 20, n_queries, 'passive_up', color='C4')

# plot_conf_int(passive_weighted_accs, 2000, 20, n_queries, 'passive_weighted', color='C4')

# plot_conf_int(passive_shuffled_accs, 2000, 20, n_queries, 'passive', color='C1')

# plot_conf_int(passive_bs_accs, 2000, 20, n_queries, 'passive_bs', color='C1')

# plot_conf_int(lc_accs, 2000, 20, n_queries, 'least_confident', color='C1')
# plot_conf_int(margin_accs, 2000, 20, n_queries, 'margin', color='C2')
# plot_conf_int(margin_sf_accs, 2000, 20, n_queries, 'margin', color='C3')
# plot_conf_int(margin_up_accs, 2000, 20, n_queries, 'margin_up', color='C5')

# plot_conf_int(margin_weighted_accs, 2000, 20, n_queries, 'margin_weighted', color='C3')

# plot_conf_int(cluster_trivial_margin_accs, 2000, 20, n_queries, 'cluster margin', color='C7')
# plot_conf_int(cluster_trivial_entropy_accs, 2000, 20, n_queries, 'cluster entropy', color='C8')

# plot_conf_int(entropy_top_accs[3], 2000, 20, n_queries, 'entropy_top_3', color='C3')
# plot_conf_int(entropy_top_accs[4], 2000, 20, n_queries, 'entropy_top_4', color='C4')
# plot_conf_int(entropy_top_accs[5], 2000, 20, n_queries, 'entropy_top_5', color='C5')

# plot_conf_int(entropy_accs, 2000, 20, n_queries, 'entropy', color='C7')
#
# plot_conf_int(diversity_img_accs, 2000, 20, n_queries, 'diversity_img', color='C9')
# plot_conf_int(diversity_txt_accs, 2000, 20, n_queries, 'diversity_txt', color='C10')
# plot_conf_int(diversity_concat_accs, 2000, 20, n_queries, 'diversity_concat', color='C7')


# plot_conf_int(cluster_concat_accs, 2000, 20, n_queries, 'cluster_concat', color='C4')
# plot_conf_int(cluster_img_accs, 2000, 20, n_queries, 'cluster_img', color='C5')
# plot_conf_int(cluster_txt_accs, 2000, 20, n_queries, 'cluster_txt', color='C6')
#
# plot_conf_int(bald_lc5_accs, 2000, 20, n_queries, 'bald_lc5', color='C4')
# plot_conf_int(bald_margin5_accs, 2000, 20, n_queries, 'bald_margin5', color='C5')
# plot_conf_int(bald_entropy3_accs, 2000, 20, n_queries, 'bald_entropy3', color='C1')

# plot_conf_int(bald_entropy5_accs, 2000, 20, n_queries, 'bald_entropy5', color='C6')
# plot_conf_int(sud1000_accs, 2000, 20, n_queries, 'sud1000', color='C7')
# plot_conf_int(sud1000_ad64_entropy_accs, 2000, 20, n_queries, 'sud', color='C8')
# plot_conf_int(sud1000_ad64_margin_accs, 2000, 20, n_queries, 'sud1000_margin_ad64', color='C9')

# plot_conf_int(bald_modal_accs, 2000, 20, n_queries, 'bald_modal_mc', color='C2')
# plot_conf_int(bald_modal_non_mc_accs, 2000, 20, n_queries, 'bald_modal_non_mc', color='C13')
# plot_conf_int(bald_trident_accs, 2000, 20, n_queries, 'bald_trident', color='C10')
# plot_conf_int(bald_trident_mc_accs, 2000, 20, n_queries, 'bald_trident_mc', color='C1')



# plot_conf_int(bald_trident_x3_accs, 2000, 20, n_queries, 'bald_trident_x3', color='C1')
# plot_conf_int(bald_trident_x5_accs, 2000, 20, n_queries, 'bald_trident_x5', color='C2')
#
# plot_conf_int(bald_trident_based_accs, 2000, 20, n_queries, 'bald_trident_based_3', color='C3')
# plot_conf_int(bald_trident_based_5_accs, 2000, 20, n_queries, 'bald_trident_based_5', color='C4')

#
# plot_conf_int(bald_margin10_accs, 2000, 20, n_queries, 'bald_margin10', color='C6')
# plot_conf_int(bald_entropy10_accs, 2000, 20, n_queries, 'bald_entropy10', color='C7')

# plot_conf_int(bald_lc5_accs, 2000, 20, n_queries, 'bald_lc5', color='C6')

# plot_conf_int(bald_entropy5_accs, 2000, 20, n_queries, 'bald_entropy5', color='C6')

#
# plot_conf_int(entropy_accs, 2000, 20, n_queries, 'entropy', color='C4')
# plot_conf_int(qbc_lc5_accs, 2000, 20, n_queries, 'qbc_lc5', color='C5')
# plot_conf_int(qbc_margin5_accs, 2000, 20, n_queries, 'qbc_margin5', color='C4')
# plot_conf_int(qbc_entropy5_accs, 2000, 20, n_queries, 'qbc_entropy_5', color='C2')

plt.xlabel('размер размеченного набора данных')
plt.ylabel('валидационная точность')
plt.legend(loc='lower right')
plt.show()
