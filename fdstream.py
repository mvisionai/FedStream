from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from microcluster import MicroClsuters
from sklearn.neighbors import NearestNeighbors
import random
import tqdm
from scipy.spatial.distance import mahalanobis
import collections
import time
import socket
from metric_learn import NCA
import argparse

hostname = str(socket.gethostname())
print("Experiment running on server "+hostname, end="\n")

def load_data(use_data=None):

    data_load = np.load('dataset/'+use_data+str('.npy'))
    scalar = MinMaxScaler()
    #data_load[:, 0:-1] = scalar.fit_transform(data_load[:, 0:-1])
    print(data_load.shape)
    return np.asarray(data_load).astype(float)


def load_initial(data,limit_size=1000):
    class_data = {}

    limit_size = int(init_percent*len(data))

    initial_load = data[0:limit_size, :]
    all_classes = np.unique(initial_load[:, zero_index_features])

    for aclass in list(all_classes):
        class_data[int(aclass)] = initial_load[initial_load[:, zero_index_features] == aclass]

    return class_data

def partition_client_class(initial_load):

    class_data = {}

    all_classes = np.unique(initial_load[:, zero_index_features])
    for aclass in list(all_classes):
        class_data[int(aclass)] = initial_load[initial_load[:, zero_index_features] == aclass]

    return  class_data




def load_stream_data(data):

    initial_size = int(init_percent*len(data))
    stream_load = data[initial_size+1:len(data)+1, :]

    print(stream_load.shape)
    return  stream_load



def mahalanobiss(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.    """
    x_minus_mu = x - np.mean(x) #change
    if not cov:
        cov = np.cov(np.transpose(x))# np.cov(x.values.T)  #change
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, np.transpose(x_minus_mu))
    return mahal.diagonal()


def mahalanobisV2(data=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.    """
    nn = NearestNeighbors(algorithm='brute',
                          metric='mahalanobis',
                          metric_params={'V': np.cov(data)})

    return nn.fit(data)


def mahalanobisV3(data=None):


    inverse=np.linalg.inv(np.cov(data))

    return   mahalanobis(data[1], data[1],inverse)

def initial_model(class_data,cluster_no,proto_data):


    microClusters=MicroClsuters()
    microClusters.zero_index_features =zero_index_features
    cluster_val=cluster_no

    #iterate over each class data
    for keys,data_clus in class_data.items():

        #check condition for microclusters
        #print("class ", keys,data_clus.shape[0])
        if data_clus.shape[0]<=cluster_val:
            cluster_val=int(data_clus.shape[0])//2

            if cluster_val==0:
                cluster_val=1

        # Create cluster for each class data using KMEANS

        kmeans = KMeans(n_clusters=cluster_val, random_state=0).fit(data_clus[:,0:zero_index_features])
        clu_center = kmeans.cluster_centers_
        clus_label = np.asarray(kmeans.labels_)

        #print("centers ",np.asarray(clu_center).shape)

        #create micro cluster for each cluster data
        for i in range(cluster_val):
            each_cluster=data_clus[ClusterIndicesNumpy(i,clus_label)][:,0:zero_index_features]
            num_points= each_cluster.shape[0]
            cluster_center=np.asarray(clu_center[i])
            #creating microcluster
            microClusters.setMicrocluster(each_cluster,int(keys),0,num_points,[])

    print("clients ", len(microClusters.getMicrocluster()))
    return microClusters

def ClusterIndicesNumpy(clustNum, labels_array): #numpy
    return np.where(labels_array == clustNum)[0]


def learn_mahalanobis_matrix(data_x, data_y):

    nca = NCA(max_iter=1000)
    data_y = data_y.astype('int')
    nca.fit(data_x, data_y)
    matrix = nca.get_mahalanobis_matrix()
    return matrix


def convert_to_numpy(data={}):

    numpy_data=[]

    for keys, data_clus in data.items():
        numpy_data.append(list(data_clus))

    return np.asarray(numpy_data)

def parallize_client_stream(client_key,client_data,data_instance,local_pros,correct_count,acc_window,weights,global_prototype,flags=0):

    local_model = local_pros.refreshMCs()
    if len(global_prototype.getMicrocluster())>0:
            #client_prototype.insertClientMC(convert_to_numpy(global_prototype.getMicrocluster()))
            #print("all",len(global_prototype.getMicrocluster()))
            pass

    max_weight=  weights.argmax(axis=0)
    numpy_convert = convert_to_numpy(local_model.getMicrocluster())
    cluster_c = numpy_convert[:, 5]

    cluster_c_label = np.where(numpy_convert[:, 4] == 1)[0]
    label_clu_cen = numpy_convert[cluster_c_label]

    data_t = client_data[client_key][data_instance][0:zero_index_features]
    class_data = int(client_data[client_key][data_instance][zero_index_features])
    currentTime =data_instance


    # print("shapes ",acc_window)
    selected_cluster = {}
    p_label = {}

    for j, kc in enumerate(k_neigbours):
        tem_center = [list(ex) for ex in np.asarray(label_clu_cen[:, 5]).tolist()]
        knn_search = NearestNeighbors(n_neighbors=kc)
        knn_search.fit(tem_center)
        neighboaurs = knn_search.kneighbors(data_t.reshape(1, -1), return_distance=False)
        neighboaurs = neighboaurs[0]

        best_clusters = label_clu_cen[neighboaurs]
        selected_cluster[j] = (best_clusters, neighboaurs)
        predicted_labels = label_clu_cen[neighboaurs][:, 3]

        p_label[j] = predicted_labels

        # inserting client single_clusters
        #local_model.insertClientMC(best_clusters)

        if acc_window.shape[1] == acc_win_max_size:
            acc_window = np.zeros((len(k_neigbours), 1))

        if j == 0:

            if acc_window.shape[1] > 1:
                eidx = acc_window.shape[1] - 1
            else:
                eidx = acc_window.shape[1]

            new_acc_adj = np.zeros((len(k_neigbours), 1))
            acc_window = np.column_stack((acc_window, new_acc_adj))

        else:
            eidx = acc_window.shape[1] - 1

        if class_data == int(predicted_labels[0]):
            acc_window[j, eidx] = 1
        else:
            acc_window[j, eidx] = 0

    weighted_cluster, cluster_indices = selected_cluster[max_weight[0]]
    weighted_label = p_label[max_weight[0]]

   #weighted label
    label_weighted= weighted_label[0]

    # data check
    nrb_labels = weighted_cluster[:, 3]
    correct_label_index = np.where(weighted_cluster[:, 3] == int(class_data))[0]
    incorrect_label_index = np.where(weighted_cluster[:, 3] != int(class_data))[0]

    incorrect_micro_index = np.asarray(cluster_indices)[incorrect_label_index].tolist()
    correct_micro_index = np.asarray(cluster_indices)[correct_label_index].tolist()

    # update of current available microclusters by index
    #print("client update ", client_key, len(local_model.getMicrocluster()))
    local_model.updateMicroClsuter(incorrect_micro_index, 7, -1)
    local_model.updateMicroClsuter(correct_micro_index, 7, 1, currentTime)

    # update model
    local_model = local_model.updateReliability(currentTime, decay_rate, wT)
    numpy_convert_2 = convert_to_numpy(local_model.getMicrocluster())
    cluster_center = numpy_convert_2[:, 5]

    neigh_search = NearestNeighbors(n_neighbors=1)
    neigh_search.fit(np.asarray(cluster_center).tolist())
    neighs = neigh_search.kneighbors(data_t.reshape(1, -1), return_distance=True)

    # picking cluster minimum cluster distance and cluster predicted
    predicted_distance = neighs[0][0][0]
    predicted_cluster = neighs[1][0][0]

    current_clus = np.asarray(local_model.getSingleMC(predicted_cluster))
    original_radius = current_clus[2]
    clus_label = current_clus[3]
    init_psd = current_clus[9]
    global_radius = original_radius

    #check label with global prototype label
    if len(global_prototype.getMicrocluster()) > 0:

        global_convert = convert_to_numpy(global_prototype.getMicrocluster())
        global_search = NearestNeighbors(n_neighbors=1)
        global_search.fit(np.asarray(global_convert[:, 5]).tolist())
        gneighboaurs = global_search.kneighbors(data_t.reshape(1, -1), return_distance=True)

        global_distance = gneighboaurs[0][0][0]
        predicted_global = gneighboaurs[1][0][0]

        current_global_clus = np.asarray(global_prototype.getSingleMC(predicted_global))
        global_radius = current_global_clus[2]
        global_label = current_global_clus[3]
        init_psd = current_global_clus[9]

        if global_distance < predicted_distance:
            label_weighted = global_label

        if global_label == class_data:
            global_prototype = global_prototype.updateSingleReliability(predicted_global,
                                                             currentTime,decay_rate,  wT)

        #delete cluster  if condition
        if (global_distance<predicted_distance) and global_label !=class_data :
            global_prototype = global_prototype.deleteMC(predicted_global)
        elif global_label !=class_data:
            #global_prototype = global_prototype.deleteMC(predicted_global)
            pass

    # get correctly predicted label
    if label_weighted == class_data:
        correct_count = correct_count + 1


    #print("Shape ",np.asarray(client_data[client_key][data_instance]).shape,np.asarray(client_data[client_key][data_instance]))
    #client prototype uodate checking
    if (predicted_distance <= original_radius and class_data == clus_label) or predicted_distance <= original_radius:
        local_model = local_model.updateMcInfo(client_data[client_key][data_instance], predicted_cluster, currentTime)

    else:
        local_model = local_model.createNewMc(client_data[client_key][data_instance], original_radius, currentTime, max_mc)

        if fed_privacy ==True:
            #transformation learn
            init_psd = learn_mahalanobis_matrix(tem_center,  label_clu_cen[:, 3])
            class_split = list([client_data[client_key][data_instance][zero_index_features]])
            data_split = client_data[client_key][data_instance][0:zero_index_features]
            transform_prototype = np.matmul(data_split, np.asarray(init_psd)) #np.dot(data_split,np.asarray(init_psd).T)
            concatenate_class = np.asarray(list(transform_prototype) + class_split)  # ,np.linalg.inv(transform_prototype)

            # remember original radius
            global_prototype = global_prototype.createNewMc(concatenate_class, global_radius, currentTime,
                                                            global_max_mc, init_psd, True)
        else:
            global_prototype = global_prototype.createNewMc(client_data[client_key][data_instance], original_radius, currentTime, global_max_mc, init_psd, False)


    weights = np.sum(acc_window, axis=1) / acc_window.shape[1]

    #global_prototype = global_prototype.globalUpdateReliability(currentTime, decay_rate, wT)


    return {client_key:[local_model,correct_count,acc_window,weights,global_prototype]}


def parallize_stream(client_key,client_data,data_instance,client_prototype,correct_count,acc_window,weights):


    max_weight=  weights.argmax(axis=0)
    numpy_convert = convert_to_numpy(client_prototype.getMicrocluster())
    cluster_c = numpy_convert[:, 5]
    cluster_c_label = np.where(numpy_convert[:, 4] == 1)[0]
    label_clu_cen = numpy_convert[cluster_c_label]

    data_t = client_data[client_key][data_instance][0:zero_index_features]
    class_data = int(client_data[client_key][data_instance][zero_index_features])
    currentTime =data_instance

    # print("shapes ",acc_window)
    selected_cluster = {}
    p_label = {}

    for j, kc in enumerate(k_neigbours):

        # print("len ",len(micro_model.getMicrocluster()))
        tem_center = [list(ex) for ex in np.asarray(label_clu_cen[:, 5]).tolist()]

        knn_search = NearestNeighbors(n_neighbors=kc)

        #NAN to zero
        if np.any(np.isnan(tem_center)):
            where_are_NaNs = np.isnan(tem_center)
            index_extract = np.where(where_are_NaNs==1)
            for r,c in zip(index_extract[0],index_extract[1]):
              tem_center[r][c]=0
        #end condition

        knn_search.fit(tem_center)
        neighboaurs = knn_search.kneighbors(data_t.reshape(1, -1), return_distance=False)
        neighboaurs = neighboaurs[0]

        best_clusters = label_clu_cen[neighboaurs]

        selected_cluster[j] = (best_clusters, neighboaurs)

        predicted_labels = label_clu_cen[neighboaurs][:, 3]

        unique_predicted = np.unique(predicted_labels)

        p_label[j] = predicted_labels

        # inserting client single_clusters
        client_prototype.insertClientMC(best_clusters)

        if acc_window.shape[1] == acc_win_max_size:
            acc_window = np.zeros((len(k_neigbours), 1))

        if j == 0:

            if acc_window.shape[1] > 1:
                eidx = acc_window.shape[1] - 1
            else:
                eidx = acc_window.shape[1]

            new_acc_adj = np.zeros((len(k_neigbours), 1))
            acc_window = np.column_stack((acc_window, new_acc_adj))

        else:
            eidx = acc_window.shape[1] - 1

        if class_data == int(predicted_labels[0]):
            acc_window[j, eidx] = 1
        else:
            acc_window[j, eidx] = 0

        # print("acc",acc_window)
    #print("My weights",max_weight)
    weighted_cluster, cluster_indices = selected_cluster[max_weight[0]]
    weighted_label = p_label[max_weight[0]]

    #weighted label
    label_weighted=weighted_label[0]

    # data check
    nrb_labels = weighted_cluster[:, 3]
    correct_label_index = np.where(weighted_cluster[:, 3] == int(class_data))[0]
    correct_label = weighted_cluster[correct_label_index]

    incorrect_label_index = np.where(weighted_cluster[:, 3] != int(class_data))[0]
    incorrect_label = weighted_cluster[incorrect_label_index]

    incorrect_micro_index = np.asarray(cluster_indices)[incorrect_label_index].tolist()
    correct_micro_index = np.asarray(cluster_indices)[correct_label_index].tolist()

    # print(incorrect_micro_index,correct_micro_index)

    # update of current available microclusters by index
    client_prototype.updateMicroClsuter(incorrect_micro_index, 7, -1)
    client_prototype.updateMicroClsuter(correct_micro_index, 7, 1, currentTime)

    # update model
    client_prototype = client_prototype.updateReliability(currentTime, decay_rate, wT)

    numpy_convert_2 = convert_to_numpy(client_prototype.getMicrocluster())
    cluster_center = numpy_convert_2[:, 5]

    neigh_search = NearestNeighbors(n_neighbors=1)
    neigh_search.fit(np.asarray(cluster_center).tolist())
    neighs = neigh_search.kneighbors(data_t.reshape(1, -1), return_distance=True)

    # picking cluster minimum cluster distance and cluster predicted
    predicted_distance = neighs[0][0][0]
    predicted_cluster = neighs[1][0][0]

    #print(neighs)

    #exit(1)
    current_clus = np.asarray(client_prototype.getSingleMC(predicted_cluster))
    original_radius = current_clus[2]
    clus_label = current_clus[3]

    # get correctly predicted label
    if label_weighted == class_data:
        correct_count = correct_count + 1

    #client prototype uodate checking
    if (predicted_distance <= original_radius and class_data == clus_label):

        client_prototype = client_prototype.updateMcInfo(client_data[client_key][data_instance], predicted_cluster, currentTime)

    else:

        #print(client_data[client_key][data_instance],original_radius)

        client_prototype = client_prototype.createNewMc(client_data[client_key][data_instance], original_radius, currentTime, max_mc)



    weights = np.sum(acc_window, axis=1) / acc_window.shape[1]

    return {client_key:[client_prototype, correct_count, acc_window, weights]}


def FederatedStream(cleints_data, proto_data):

    correct_count = 0
    #initializing client prototype
    client_prototypes = collections.OrderedDict()

    global_iterate = collections.OrderedDict()
    csv_client_keys = []

    for cl_key in cleints_data.keys():
        client_prototypes[cl_key] = initial_model(partition_client_class(proto_data[cl_key]), fed_number_clusters, proto_data[cl_key])
        csv_client_keys.append(cl_key)


    micro_model = MicroClsuters()
    micro_model.emptyMicrocluster()
    micro_model.zero_index_features = zero_index_features

    acc_window = np.zeros((len(k_neigbours), 1))
    weights = np.ones((len(k_neigbours), 1))
    flags = 0
    counter_flag = 0
    #looping through clients

    accuracy_step_list = []
    runtime_file = open("results/fed/"+hostname+"_"+dataset+"_"+str(client_no)+str(max_mc) + "_2local_fedstream_runtime.txt", "a+")
    continue_status = False

    for i in tqdm.tqdm(range(len(cleints_data['c_1']))):
        if counter_flag == 0:
         start_t = time.time()

        for keys, clt_data in cleints_data.items():

            # initialization for each client
            # looping through client data
            client_prototype=client_prototypes[keys]

            if i > 0:
                select_instance = global_iterate[keys+'_'+str(i-1)]
                client_prototype = select_instance[0]
                correct_count = select_instance[1]
                acc_window = select_instance[2]
                weights = np.asarray([select_instance[3]])

            try:
                error_check = cleints_data[keys][i]
            except Exception:
                print("Running Experiment Ended", end="\n")
                continue_status = True
                continue

            fun_ret = parallize_client_stream(keys, cleints_data, i, client_prototype, correct_count, acc_window,
                                              weights, micro_model, flags)
            global_iterate[keys + '_' + str(i)] = fun_ret[keys]
            micro_model = fun_ret[keys][4]

        if continue_status:
            continue


        counter_flag=counter_flag+1
        if (i + 1) % round_time == 0:

            counter_flag = 0
            local_accuracies = []
            print("\n")
            for skeys in cleints_data.keys():
              accuracy_select=global_iterate[skeys+'_'+str(i-1)]
              current_acc= round((accuracy_select[1] / (i + 1)) * 100, 3)
              print("Client-{} Streamed {} data samples with accuracy : {}%".format(skeys, i + 1, current_acc),end="\n")
              local_accuracies.append(current_acc)

            #saving results
            accuracy_step_list.append([i + 1, *local_accuracies])
            df = pd.DataFrame(accuracy_step_list, columns=['step', *csv_client_keys])
            df.to_csv("results/fed/"+hostname+"_"+dataset+"_"+str(client_no)+"_"+str(max_mc)+"_2local_fedstream.csv", index=False)


            end_time = time.time()
            seconds_calculate = end_time-start_t
            minutes = int((seconds_calculate)//60)
            seconds = int(seconds_calculate%60)
            print("Execution time for "+str(i + 1)+" ", str(minutes)+":"+str(seconds)+" Global Instances : "+str(micro_model.getClusInstances()), end="\n")

            runtime_file.write(str(i+1) + " " + str(seconds_calculate) + "\n")

    return convert_to_numpy(micro_model.getMicrocluster())



def ClientStream(cleints_data,proto_data):

    correct_count = 0
    #initializing client prototype
    client_prototypes = collections.OrderedDict()

    global_iterate = collections.OrderedDict()

    csv_client_keys = []

    for cl_key in cleints_data.keys():
        client_prototypes[cl_key] = initial_model(partition_client_class(proto_data[cl_key]),fed_number_clusters,[])
        csv_client_keys.append(cl_key)


    acc_window = np.zeros((len(k_neigbours), 1))
    weights = np.ones((len(k_neigbours), 1))
    flags=0
    counter_flag = 0
    #looping through clients



    accuracy_step_list=[]
    runtime_file = open("results/client/"+hostname + "_" + dataset + "_" + str(client_no) + "_client_runtime.txt", "a+")
    continue_status = False

    for i in tqdm.tqdm(range(len(cleints_data['c_1']))):


        if counter_flag == 0:
         start_t=time.time()

        for keys, clt_data in cleints_data.items():

            # initialization for each client
            # looping through client data
            client_prototype=client_prototypes[keys]

            if i > 0:
                select_instance = global_iterate[keys+'_'+str(i-1)]
                client_prototype = select_instance[0]
                correct_count = select_instance[1]
                acc_window = select_instance[2]
                weights=np.asarray([select_instance[3]])

            try:

                error_check = cleints_data[keys][i]

            except Exception:
                print("Running Experiment Ended",end="\n")
                continue_status = True
                continue

            fun_ret=parallize_stream(keys, cleints_data, i, client_prototype, correct_count, acc_window,
                                    weights)

            global_iterate[keys+'_'+str(i)]=fun_ret[keys]

            if continue_status:
                continue

        counter_flag=counter_flag+1
        #print("flag ",counter_flag)
        #print results
        if (i + 1) % round_time == 0:

            counter_flag=0

            local_accuracies=[]
            print("\n")
            for  skeys in  cleints_data.keys():
              accuracy_select=global_iterate[skeys+'_'+str(i-1)]
              current_acc= round((accuracy_select[1] / (i + 1)) * 100,3)
              print("Client-{} Streamed {} data samples with accuracy : {}%".format(skeys, i + 1, current_acc),end="\n")
              local_accuracies.append(current_acc)

            #saving results
            accuracy_step_list.append([i + 1,*local_accuracies])
            df = pd.DataFrame(accuracy_step_list, columns=['step', *csv_client_keys])
            df.to_csv("results/client/"+hostname+"_"+dataset+"_"+str(client_no)+"_clientstream.csv",index=False)


            end_time = time.time()
            seconds_calculate=end_time-start_t
            minutes=int((seconds_calculate)//60)
            seconds=int(seconds_calculate%60)
            print("Execution time for "+str(i + 1)+" ",str(minutes)+":"+str(seconds),end="\n")
            runtime_file.write(str(i + 1) + " " + str(seconds_calculate) + "\n")

def ClientsStream(cleints_data,micro_model):
    client_accuracy_list={}
    correct_count=0


    #initializing client prototype
    client_prototypes={}

    for cl_key in  cleints_data.keys():
        client_prototypes[cl_key]=MicroClsuters()
        client_prototypes[cl_key].insertClientMC(convert_to_numpy(micro_model.getMicrocluster()))

    micro_model.emptyMicrocluster()


    #looping through clients
    for keys, clt_data in cleints_data.items():

        acc_list=[]
        #initialization for each client
        acc_window = np.zeros((len(k_neigbours), 1))
        weights = np.ones((len(k_neigbours), 1))
        max_weight = weights.argmax(axis=0)

        #looping through client data
        for i,np_d in enumerate(clt_data):

              client_prototype = client_prototypes[keys]
              numpy_convert = convert_to_numpy(client_prototype.getMicrocluster())
              cluster_c = numpy_convert[:, 5]
              cluster_c_label = np.where(numpy_convert[:, 4] == 1)[0]
              label_clu_cen = numpy_convert[cluster_c_label]

              data_t=np_d[0:zero_index_features]
              class_data=int(np_d[zero_index_features])
              currentTime=i


              #print("shapes ",acc_window)

              selected_cluster = {}
              p_label = {}

              for j,kc in enumerate(k_neigbours):

                  #print("len ",len(micro_model.getMicrocluster()))
                  tem_center=[list(ex) for ex in np.asarray(label_clu_cen[:, 5]).tolist()]

                  knn_search = NearestNeighbors(n_neighbors=kc)
                  knn_search.fit(tem_center)
                  neighboaurs=knn_search.kneighbors(data_t.reshape(1,-1), return_distance=False)
                  neighboaurs=neighboaurs[0]

                  best_clusters=label_clu_cen[neighboaurs]

                  selected_cluster[j] =(best_clusters,neighboaurs)


                  predicted_labels=label_clu_cen[neighboaurs][:,3]

                  unique_predicted=np.unique(predicted_labels)


                  p_label[j]=predicted_labels


                  #inserting client single_clusters
                  client_prototype.insertClientMC(best_clusters)


                  if acc_window.shape[1] == acc_win_max_size:
                      acc_window = np.zeros((len(k_neigbours),1))


                  if j==0 :

                    if acc_window.shape[1]>1:
                        eidx = acc_window.shape[1]-1
                    else:
                      eidx = acc_window.shape[1]

                    new_acc_adj = np.zeros((len(k_neigbours), 1))
                    acc_window = np.column_stack((acc_window, new_acc_adj))

                  else:
                      eidx = acc_window.shape[1]-1


                  if class_data == int(predicted_labels[0]):
                      acc_window[j, eidx] = 1
                  else:
                      acc_window[j, eidx] = 0

                  #print("acc",acc_window)

              weighted_cluster,cluster_indices = selected_cluster[max_weight[0]]
              weighted_label = p_label[max_weight[0]]


              #get correctly predicted label
              if weighted_label[0] == class_data:
                    correct_count = correct_count + 1



              #data check
              nrb_labels = weighted_cluster[:,3]
              correct_label_index =np.where(weighted_cluster[:, 3] == int(class_data))[0]
              correct_label = weighted_cluster[correct_label_index]

              incorrect_label_index = np.where(weighted_cluster[:, 3] != int(class_data))[0]
              incorrect_label = weighted_cluster[incorrect_label_index]


              incorrect_micro_index=np.asarray(cluster_indices)[incorrect_label_index].tolist()
              correct_micro_index=np.asarray(cluster_indices)[correct_label_index].tolist()

              #print(incorrect_micro_index,correct_micro_index)

              #update of current available microclusters by index
              client_prototype.updateMicroClsuter(incorrect_micro_index,7,-1)
              client_prototype.updateMicroClsuter(correct_micro_index, 7, 1,currentTime)


              #update model
              client_prototype  = client_prototype.updateReliability(currentTime,decay_rate,wT)

              numpy_convert_2 = convert_to_numpy(client_prototype.getMicrocluster())
              cluster_center = numpy_convert_2[:, 5]

              neigh_search = NearestNeighbors(n_neighbors=1)
              neigh_search.fit(np.asarray(cluster_center).tolist())
              neighs = neigh_search.kneighbors(data_t.reshape(1, -1), return_distance=True)

              #picking cluster minimum cluster distance and cluster predicted
              predicted_distance = neighs[0][0][0]
              predicted_cluster=neighs[1][0][0]


              current_clus=np.asarray(client_prototype.getSingleMC(predicted_cluster))
              original_radius = current_clus[2]
              clus_label=current_clus[3]

              if (predicted_distance <=original_radius  and  class_data == clus_label) :

                  client_prototype = client_prototype.updateMcInfo(np_d,predicted_cluster,currentTime)

                  #create global prototype or cluster
              else:

                  client_prototype = client_prototype.createNewMc(np_d,original_radius,currentTime,max_mc)

              #Accuracy Calculation
              current_acc = (correct_count / (i + 1)) * 100
              if (i+1)%1000 == 0:
                  print("Clien-{} Streamed {} data samples with accuracy : {}%".format(keys,i + 1, current_acc))

              weights = np.sum(acc_window,axis=1)/acc_window.shape[1]

        client_accuracy_list[keys]=current_acc
        correct_count=0

    print("All clients accuaracy ",client_accuracy_list)


    return convert_to_numpy(micro_model.getMicrocluster())


def StreamLearning(data,micro_model):

    global acc_window
    global weights



    #max_weight=np.max(weights)
    max_weight=weights.argmax(axis=0)

    accuracy_list=[]
    correct_count=0

    accuracy_step_list=[]

    counter_flag = 0

    runtime_file = open("results/single/" + hostname + "_" + dataset +"_"+str(client_no)+"_single_runtime.txt", "a+")

    for i,np_d in enumerate(data):

          if counter_flag == 0:
            start_t = time.time()
         # get center clusters bases on cluster flage
          numpy_convert = convert_to_numpy(micro_model.getMicrocluster())
          cluster_c = numpy_convert[:, 5]
          cluster_c_label = np.where(numpy_convert[:, 4] == 1)[0]
          label_clu_cen = numpy_convert[cluster_c_label]

          data_t=np_d[0:zero_index_features]
          class_data=int(np_d[zero_index_features])
          currentTime=i


          #print("shapes ",acc_window)

          selected_cluster = {}
          p_label = {}



          for j,kc in enumerate(k_neigbours):


              knn_search = NearestNeighbors(n_neighbors=kc)
              knn_search.fit(np.asarray(label_clu_cen[:,5]).tolist())
              neighboaurs=knn_search.kneighbors(data_t.reshape(1,-1), return_distance=False)
              neighboaurs=neighboaurs[0]

              #print("My knn ",kc,i)

              selected_cluster[j] =(label_clu_cen[neighboaurs],neighboaurs)


              predicted_labels=label_clu_cen[neighboaurs][:,3]

              unique_predicted=np.unique(predicted_labels)

              #print("predicted unique",unique_predicted)

              p_label[j]=predicted_labels
              if acc_window.shape[1] == acc_win_max_size:
                  acc_window = np.zeros((len(k_neigbours),1))


              if j==0 :

                if acc_window.shape[1]>1:
                    eidx = acc_window.shape[1]-1
                else:
                  eidx = acc_window.shape[1]

                new_acc_adj = np.zeros((len(k_neigbours), 1))
                acc_window = np.column_stack((acc_window, new_acc_adj))

              else:
                  eidx = acc_window.shape[1]-1

                  #print("idx",eidx)

              if class_data == int(predicted_labels[0]):
                  acc_window[j, eidx] = 1
              else:
                  acc_window[j, eidx] = 0
              #print("acc",acc_window)


          weighted_cluster,cluster_indices = selected_cluster[max_weight[0]]
          weighted_label = p_label[max_weight[0]]


          #get correctly predicted label
          if weighted_label[0] == class_data:
                correct_count = correct_count + 1


          #data check
          nrb_labels = weighted_cluster[:,3]
          correct_label_index =np.where(weighted_cluster[:, 3] == int(class_data))[0]
          correct_label = weighted_cluster[correct_label_index]

          incorrect_label_index = np.where(weighted_cluster[:, 3] != int(class_data))[0]
          incorrect_label = weighted_cluster[incorrect_label_index]


          incorrect_micro_index=np.asarray(cluster_indices)[incorrect_label_index].tolist()
          correct_micro_index=np.asarray(cluster_indices)[correct_label_index].tolist()

          #print(incorrect_micro_index,correct_micro_index)

          #update of current available microclusters by index
          micro_model.updateMicroClsuter(incorrect_micro_index,7,-1)
          micro_model.updateMicroClsuter(correct_micro_index, 7, 1,currentTime)

          #update model
          micro_model  = micro_model.updateReliability(currentTime,decay_rate,wT)
          numpy_convert_2 = convert_to_numpy(micro_model.getMicrocluster())
          cluster_center = numpy_convert_2[:, 5]

          neigh_search = NearestNeighbors(n_neighbors=1)
          neigh_search.fit(np.asarray(cluster_center).tolist())
          neighs = neigh_search.kneighbors(data_t.reshape(1, -1), return_distance=True)

          #picking cluster minimum cluster distance and cluster predicted
          predicted_distance = neighs[0][0][0]
          predicted_cluster=neighs[1][0][0]


          current_clus=np.asarray(micro_model.getSingleMC(predicted_cluster))
          original_radius = current_clus[2]
          clus_label=current_clus[3]

          if (predicted_distance <=original_radius  and  class_data == clus_label) :

              micro_model = micro_model.updateMcInfo(np_d,predicted_cluster,currentTime)
          else:

              micro_model = micro_model.createNewMc(np_d,original_radius,currentTime,max_mc)


          #Accuracy Calculation
          current_acc = round((correct_count / (i + 1)) * 100,3)
          if (i+1)%round_time == 0:

              counter_flag = 0
              accuracy_step_list.append([i+1,current_acc])
              df = pd.DataFrame(accuracy_step_list, columns=['step', "accuracy"])
              df.to_csv("results/single/"+hostname+"_"+dataset+"_"+str(client_no)+"_singlestream.csv", index=False)

              print("\n Streamed {} data samples with accuracy : {}%".format(i + 1, current_acc))
              end_time = time.time()
              seconds_calculate = end_time - start_t
              minutes = int((seconds_calculate) // 60)
              seconds = int(seconds_calculate % 60)
              print("Execution time for " + str(i + 1) + " ",
                    str(minutes) + ":" + str(seconds) + " Global Instances : " + str(micro_model.getClusInstances()),
                    end="\n")
              runtime_file.write(str(i + 1) + " " + str(seconds_calculate) + "\n")

          weights = np.sum(acc_window, axis=1)/acc_window.shape[1]




    return convert_to_numpy(micro_model.getMicrocluster())


def classify():
    pass


def ClientData(data,num_clients=10,client_initial="c"):
    # create a list of client names

    order_dict = collections.OrderedDict()
    client_names = ['{}_{}'.format(client_initial, i + 1) for i in range(num_clients)]

    # randomize the data
    random.shuffle(data)

    # shard data and place at each client
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    return {client_names[i]: shards[i] for i in range(len(client_names))},client_names

def StreamClientData(data,num_clients=10,client_initial="c"):
    # create a list of client names

    order_dict = collections.OrderedDict()
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


def clientProStreams(cleints_data):
    proto_data = collections.OrderedDict()
    stream_data = collections.OrderedDict()
    for cl_key in cleints_data.keys():

        client_records = len(cleints_data[cl_key])
        client_init = int(init_percent * client_records)
        proto_data[cl_key] = cleints_data[cl_key][0:client_init, :]
        stream_data[cl_key] = cleints_data[cl_key][client_init+1:client_records+1, :]

    return stream_data, proto_data


def yieldClientData(client_data):
    yield client_data

def run(run_type="FedStream"):

    # load dataset
    data_load = load_data(use_data=dataset)

    if (data_load[:, -1] == 0).any():
        print('Okay')
    else:
        print('Label Transformation')
        data_load[:, -1] = data_load[:, -1] - 1


    print(np.unique(data_load[:,-1]))

    if run_type == "FedStream":

        #client stream
        data_client, client_name = StreamClientData(data_load,client_no)

        # get initial prototype data and each data
        stream_data, proto_data = clientProStreams(data_client)
        # client stream with fed
        fed_learning = FederatedStream(stream_data, proto_data)

    elif run_type == "ClientsStream":

        # load initial data
        # client stream
        data_client, client_name = StreamClientData(data_load, client_no)

        # get initial prototype data and each data
        stream_data, proto_data = clientProStreams(data_client)

        #individual client stream without fed
        clients_stream = ClientStream(stream_data,proto_data)

    elif run_type == "SingleStream":

        # load initial data
        initial_load = load_initial(data_load)

        # create initial micro-cluster prototype
        model_init = initial_model(initial_load, number_clusters, None)

        # load stream data
        stream_data = load_stream_data(data_load)

        #stream learning
        learning_data = StreamLearning(stream_data, model_init)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='covtype', help="name of dataset")
    parser.add_argument('--clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--max_mc', type=int, default=200, help='max client micro-cluster')
    parser.add_argument('--global_mc', type=int, default=1000, help='max global micro-cluster')
    parser.add_argument('--features', type=int, default=54, help='Number of dataset features')
    parser.add_argument('--decay_rate', type=float, default=0.00002, help='Decay rate')
    parser.add_argument('--weight_const', type=float, default=0.6, help='Weight threshold constant')
    parser.add_argument('--local_init', type=int, default=50, help='Local initial cluster for single train')
    parser.add_argument('--global_init', type=int, default=50, help='global initial cluster for fed train')
    parser.add_argument('--reporting_interval', type=int, default=1000, help='global initial cluster for fed train')
    parser.add_argument('--initial_stream_size', type=int, default=1000, help='initial data instance size')
    parser.add_argument('--client_initial_size', type=int, default=500, help='initial data instance size for clients')
    parser.add_argument('--percent_init', type=float, default=0.1, help='set initial cluster number with percentage')
    parser.add_argument('--enable_fed_privacy', type=str2bool, default=False, const=True, nargs='?', help='Enable privacy for training')
    parser.add_argument('--run_type', choices=['FedStream', 'SingleStream', 'ClientsStream'], default='FedStream', help='experiment')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    arg = args_parser()

    init_percent = arg.percent_init
    initial_size = arg.initial_stream_size
    client_init = arg.client_initial_size
    number_clusters = arg.local_init
    fed_number_clusters = arg.local_init
    acc_win_max_size = 100
    k_neigbours = [1]
    decay_rate = arg.decay_rate
    wT = arg.weight_const
    round_time = arg.reporting_interval
    max_mc = arg.max_mc  # use 1000 for (client and single)  stream and 100 for fedstream
    global_max_mc = arg.global_mc
    acc_window = np.zeros((len(k_neigbours), 1))
    weights = np.ones((len(k_neigbours), 1))
    client_no = arg.clients
    dataset = arg.dataset
    fed_privacy = arg.enable_fed_privacy
    zero_index_features = arg.features

    print('max local: ', max_mc)
    run(run_type="FedStream")

















