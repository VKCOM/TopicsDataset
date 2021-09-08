import pickle

import matplotlib.pyplot as plt
from visualization.plots import plot_conf_int

passive_accs = []

lc_accs = []
margin_accs = []
entropy_accs = []

qbc_lc5_accs = []
qbc_margin5_accs = []
qbc_entropy5_accs = []

bald_lc3_accs = []
bald_margin3_accs = []
bald_entropy3_accs = []

bald_lc5_accs = []
bald_margin5_accs = []
bald_entropy5_accs = []

bald_margin10_accs = []
bald_entropy10_accs = []

bald_modal_accs = []
bald_modal_non_mc_accs = []

sud1000_accs = []
sud1000_ad64_entropy_accs = []
sud1000_ad64_margin_accs = []
sud1000_ad64_margin_no_mult_accs = []
sud1000_ad64_margin_sparse_accs = []

sud_top20_accs = []
sud_top50_accs = []
sud_top100_accs = []
sud_top1000_accs = []


for i in range(1, 5):
    state = pickle.load(open('statistic/topics/torch/d128/passive_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    passive_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/least_confident_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    lc_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/margin_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    margin_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/entropy_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    entropy_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch/d128/qbc_lc5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # qbc_lc5_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/qbc_margin5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # qbc_margin5_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch/d128/qbc_entropy5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # qbc_entropy5_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch/d128/bald_lc3_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_lc3_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_margin3_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_margin3_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_entropy3_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_entropy3_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_lc5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_lc5_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_margin5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_margin5_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_entropy5_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_entropy5_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch/d128/bald_margin10_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    # bald_margin10_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_entropy10_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_entropy10_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_modal_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_modal_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/bald_modal_non_mc_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    bald_modal_non_mc_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/sud1000_trivial_encode_entropy_i2000_b20_q100_' + str(i) + '.pkl', 'rb'))
    sud1000_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/sud1000_trivial_encode_ad64_entropy_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    sud1000_ad64_entropy_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/sud1000_trivial_encode_ad64_margin_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    sud1000_ad64_margin_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/sud1000_trivial_encode_ad64_margin_no_mult_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    sud1000_ad64_margin_no_mult_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch/d128/sud1000_trivial_encode_ad64_margin_sparse_i2000_b20_q100_sf512_' + str(i) + '.pkl', 'rb'))
    sud1000_ad64_margin_sparse_accs.append(state['performance_history'])

    state = pickle.load(open(
        'statistic/topics/torch/d128/sud_top20_trivial_encode_ad64_margin_i2000_b20_q100_sf512_' + str(i) + '.pkl',
        'rb'))
    sud_top20_accs.append(state['performance_history'])

    state = pickle.load(open(
        'statistic/topics/torch/d128/sud_top50_trivial_encode_ad64_margin_i2000_b20_q100_sf512_' + str(i) + '.pkl',
        'rb'))
    sud_top50_accs.append(state['performance_history'])

    state = pickle.load(open(
        'statistic/topics/torch/d128/sud_top100_trivial_encode_ad64_margin_i2000_b20_q100_sf512_' + str(i) + '.pkl',
        'rb'))
    sud_top100_accs.append(state['performance_history'])

    state = pickle.load(open(
        'statistic/topics/torch/d128/sud_top1000_trivial_encode_ad64_margin_i2000_b20_q100_sf512_' + str(i) + '.pkl',
        'rb'))
    sud_top1000_accs.append(state['performance_history'])

n_instances = state['n_instances']
n_queries = state['n_queries']

plot_conf_int(passive_accs, 2000, 20, n_queries, 'passive', color='C0')
# plot_conf_int(lc_accs, 2000, 20, n_queries, 'least confident', color='C1')
# plot_conf_int(margin_accs, 2000, 20, n_queries, 'margin', color='C2')
# plot_conf_int(entropy_accs, 2000, 20, n_queries, 'entropy', color='C3')

# plot_conf_int(qbc_lc5_accs, 2000, 20, n_queries, 'qbc_lc', color='C4')
# plot_conf_int(qbc_margin5_accs, 2000, 20, n_queries, 'qbc_margin', color='C5')
# plot_conf_int(qbc_entropy5_accs, 2000, 20, n_queries, 'qbc_entropy', color='C6')

# plot_conf_int(bald_lc3_accs, 2000, 20, n_queries, 'bald_lc3', color='C7')
# plot_conf_int(bald_margin3_accs, 2000, 20, n_queries, 'bald_margin3', color='C8')
# plot_conf_int(bald_entropy3_accs, 2000, 20, n_queries, 'bald_entropy3', color='C9')

# plot_conf_int(bald_lc5_accs, 2000, 20, n_queries, 'bald_lc', color='C7')
# plot_conf_int(bald_margin5_accs, 2000, 20, n_queries, 'bald_margin', color='C8')
# plot_conf_int(bald_entropy5_accs, 2000, 20, n_queries, 'bald_entropy', color='C9')

# plot_conf_int(bald_margin10_accs, 2000, 20, n_queries, 'bald_margin10', color='C13')
# plot_conf_int(bald_entropy10_accs, 2000, 20, n_queries, 'bald_entropy10', color='C14')

# plot_conf_int(bald_modal_accs, 2000, 20, n_queries, 'bald_modal_mc', color='C15')
# plot_conf_int(bald_modal_non_mc_accs, 2000, 20, n_queries, 'bald_modal_non_mc', color='C16')

# plot_conf_int(sud1000_accs, 2000, 20, n_queries, 'sud1000', color='C17')
# plot_conf_int(sud1000_ad64_entropy_accs, 2000, 20, n_queries, 'sud1000_entropy_ad64', color='C18')
# plot_conf_int(sud1000_ad64_margin_accs, 2000, 20, n_queries, 'sud1000_margin_ad64', color='C19')
# plot_conf_int(sud1000_ad64_margin_no_mult_accs, 2000, 20, n_queries, 'sud1000_margin_no_mult_ad64', color='C20')
# plot_conf_int(sud1000_ad64_margin_sparse_accs, 2000, 20, n_queries, 'sud1000_margin_sparse_ad64', color='C21')

# plot_conf_int(sud_top20_accs, 2000, 20, n_queries, 'sud_top20', color='C21')
# plot_conf_int(sud_top50_accs, 2000, 20, n_queries, 'sud_top50', color='C23')
# plot_conf_int(sud_top100_accs, 2000, 20, n_queries, 'sud_top100', color='C24')
# plot_conf_int(sud_top1000_accs, 2000, 20, n_queries, 'sud_top1000', color='C25')


# plot_conf_int(cluster_trivial_margin_accs, 2000, 20, n_queries, 'cluster margin', color='C7')
# plot_conf_int(cluster_trivial_entropy_accs, 2000, 20, n_queries, 'cluster entropy', color='C8')
# plot_conf_int(cluster_concat_accs, 2000, 20, n_queries, 'cluster_concat', color='C4')
# plot_conf_int(cluster_img_accs, 2000, 20, n_queries, 'cluster_img', color='C5')
# plot_conf_int(cluster_txt_accs, 2000, 20, n_queries, 'cluster_txt', color='C6')

plt.xlabel('labeled size')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()
