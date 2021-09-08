import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

state = pickle.load(open('statistic/passive_ll_b100_2.pkl', 'rb'))
state_2 = pickle.load(open('statistic/learning_loss_i100_b100_2.pkl', 'rb'))

init_size = state['init_size']
n_queries = state['n_queries']
batch_size = 100

plt.plot(range(init_size, init_size + n_queries * batch_size + 1, batch_size), state['performance_history'], label='random')
plt.plot(range(init_size, init_size + n_queries * batch_size + 1, batch_size), state_2['performance_history'])

plt.xlabel('labeled size')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()
#
# state = pickle.load(open('statistic/ll_w64_i100_b100_2.pkl', 'rb'))
# print(state['loss_history'])
# tf.print(state['learning_loss_history'])
# plt.plot(range(len(state['loss_history'])), state['loss_history'])
# plt.plot(range(len(state['learning_loss_history'])), state['learning_loss_history'])
# plt.show()

