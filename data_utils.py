import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import collections

def StreamClientData(data,num_clients=10,client_initial="c"):
    # create a list of client names

    order_dict=collections.OrderedDict()
    client_names = ['{}_{}'.format(client_initial, i + 1) for i in range(num_clients)]

    # shard data and place at each client
    shards = [[] for i in range(0,num_clients)]

    #new client partition
    counter_check=0

    for i in range(0, len(data), 1):

        shards[counter_check].append(data[i])
        counter_check = counter_check + 1

        if (i+1)%num_clients == 0:
          counter_check=0

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    for i in range(len(client_names)):
        order_dict[client_names[i]]=np.asarray(shards[i])

    return order_dict,client_names


def load_data(use_data=None):

    data_load = np.load('dataset/'+use_data+str('.npy'))
    scalar = MinMaxScaler()
    #data_load[:, 0:-1] = scalar.fit_transform(data_load[:, 0:-1])
    print(data_load.shape)

    return  np.asarray(data_load).astype(np.float)

