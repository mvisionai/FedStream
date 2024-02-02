"""
Microcluster Class.

A class for creating and maintaining micro-clusters.
"""
import numpy as np
from scipy.spatial.distance import cdist
import warnings
import collections


class MicroClsuters:

    def __init__(self, data=None, label=None, extime=0, data_pt=0):

        self.data = data
        self.extime = extime
        self.label = label
        self.data_pt = data_pt
        self.microclusters={}
        self.zero_index_features = 0

    def getClusInstances(self):

       return  len(self.microclusters)

    def getMicrocluster(self):
        return self.microclusters



    def getClusterLabelCondition(self):

        numpy_data=[]

        for keys in self.getMicrocluster().copy():

            numpy_data.append(list(self.microclusters[keys]))

        new_data = np.asarray(numpy_data)

        labels_data = new_data[:,3]

        #max_val=max(labels_data.tolist())

        occurs=collections.Counter(labels_data)
        max_occur=occurs.most_common(1)[0][0]

        max_occur_index = np.where(new_data[:, 3] == int(max_occur))[0]
        max_occur_clust = new_data[max_occur_index]
        return  max_occur_index,max_occur_clust



    def setMicrocluster(self,data,label,extime,data_pt,psd=[]):

        self.data = data
        self.extime = extime
        self.data_pt = data_pt
        self.label = label

        #print("label ", self.data)
        LS=np.sum(self.data,axis=0)
        SS = np.sum(np.square(self.data),axis=0)
        label=self.label
        label_flag=1
        mc_center=LS/ self.data_pt
        psd_matrix = psd
        warnings.filterwarnings('ignore')
        mc_radius=  np.sqrt(np.sum(SS /self.data_pt) - np.sum(np.square((LS / self.data_pt))))
        mc_time=self.extime
        mc_importance=1
        no_Instances=self.getClusInstances()+1
        self.microclusters[no_Instances] = [LS, SS, mc_radius, label, label_flag, mc_center, mc_time, mc_importance, data_pt, psd_matrix]

    def mergeMC(self, cluster_index, cluster_s, psd=[]):

        clsuter_np = np.asarray(cluster_s[:, 5].tolist())
        D = cdist(clsuter_np, clsuter_np)

        # set zero values to 1000
        D[D == 0] = 1000
        min_value = np.min(D)
        sorted_vals = np.where(D == min_value)
        row_index = sorted_vals[0][0]
        column_index = sorted_vals[1][0]
        ##max_occur_clust = new_data[min_mc_ind]

        micro_1_select = row_index
        micro_2_select = column_index

        micro_1_map = cluster_index[micro_1_select]
        micro_2_map = cluster_index[micro_2_select]

        first_mc = self.getSingleMC(micro_1_map)
        second_mc = self.getSingleMC(micro_2_map)

        no_Instances = self.getClusInstances() + 1

        LS = np.add(first_mc[0], second_mc[0])
        SS = np.add(first_mc[1], second_mc[1])
        N_pt = first_mc[8] + second_mc[8]
        label = first_mc[3]
        label_flag = 1
        mc_center = LS / N_pt
        mc_radius = np.sqrt(np.sum(SS / N_pt) - np.sum(np.square((LS / N_pt))))

        mc_time = max(first_mc[6], second_mc[6])
        mc_importance = max(first_mc[7], second_mc[7])

        self.microclusters[no_Instances] = [LS, SS, mc_radius, label, label_flag, mc_center, mc_time, mc_importance, 1,
                                            psd]


        ignore_list = [micro_1_map, micro_2_map]

        for clus_in in cluster_index:
            if clus_in not in ignore_list:
                self.microclusters.pop(clus_in + 1)

        # reshuffle microclsuters keys
        new_instance_cluster = {}
        for index, keys in enumerate(self.getMicrocluster().copy()):
            new_instance_cluster[index + 1] = self.microclusters[keys]

        self.microclusters = new_instance_cluster

        return self

    def mergeMC2(self,cluster_index,cluster_s,psd=[]):

        clsuter_np=np.asarray(cluster_s[:,5].tolist())
        D=cdist(clsuter_np,clsuter_np)


        #set zero values to 1000

        D[D==0]=1000
        min_value=np.min(D,axis=0)

        min_mc_ind = np.where(D==min_value)[0]
        ##max_occur_clust = new_data[min_mc_ind]

        micro_1_select=min_mc_ind[0]
        micro_2_select=min_mc_ind[2]

        micro_1_map=cluster_index[micro_1_select]
        micro_2_map=cluster_index[micro_2_select]

        first_mc = self.getSingleMC(micro_1_map+1)
        second_mc= self.getSingleMC(micro_2_map+1)

        no_Instances = self.getClusInstances() + 1

        LS=np.add(first_mc[0],second_mc[0])
        SS=np.add(first_mc[1],second_mc[1])
        N_pt=first_mc[8]+second_mc[8]
        label = first_mc[3]
        label_flag = 1
        mc_center = LS/N_pt
        mc_radius = np.sqrt(np.sum(SS/N_pt) - np.sum(np.square((LS/N_pt))))

        mc_time = max(first_mc[6],second_mc[6])
        mc_importance = max(first_mc[7],second_mc[7])

        self.microclusters[no_Instances] = [LS, SS, mc_radius, label, label_flag, mc_center, mc_time, mc_importance, 1,psd]

        ignore_list=[micro_1_map,micro_2_map]

        for clus_in in cluster_index:
            if clus_in not in ignore_list:
                self.microclusters.pop(clus_in + 1)

        # reshuffle microclsuters keys
        new_instance_cluster = {}
        for index, keys in enumerate(self.getMicrocluster().copy()):
            new_instance_cluster[index + 1] = self.microclusters[keys]

        self.microclusters = new_instance_cluster

        return  self


    def createNewMc(self,data,radius,extime,clusterLimit,psd=[],gtype=False):

        if gtype == True:
            #Get transform data back to original space
           data_split = data[0:self.zero_index_features]
           class_split = int(data[self.zero_index_features])
           data_tr = np.linalg.pinv(np.asarray(psd)) @ data_split
           data = np.asarray(list(data_tr) + [class_split])


        if len(self.getMicrocluster()) > clusterLimit:
            cluster_index,micro_clus= self.getClusterLabelCondition()
            self.mergeMC(cluster_index,micro_clus,psd)


        data_t = data[0:self.zero_index_features]
        class_data = int(data[self.zero_index_features])

        self.data = data_t
        self.extime = extime
        self.label = class_data
        self.data_pt=1

        if len(list(np.asarray(data_t).shape))==1:

            LS = self.data
            SS = np.square(self.data)
        else:
            LS = np.sum(self.data, axis=0)
            SS = np.sum(np.square(self.data))

        label = self.label
        label_flag = 1
        mc_center = LS
        mc_radius = radius
        mc_time = self.extime
        mc_importance = 1
        no_Instances = self.getClusInstances()+1
        self.microclusters[no_Instances] = [LS, SS, mc_radius, label, label_flag, mc_center, mc_time, mc_importance, 1, psd]
        return self

    def updateMicroClsuter(self,cluster_index,data_index,data,ctime=0):

        for clus in cluster_index:

                if data==1:

                   self.microclusters[clus+1][data_index] = self.microclusters[clus+1][data_index]+ data
                   self.microclusters[clus + 1][6] = ctime

                else:
                    self.microclusters[clus + 1][data_index] = self.microclusters[clus+1][data_index]+ data
                    #self.microclusters[clus + 1][6] = ctime

        return  self.microclusters


    def getSingleMC(self,index):

        return self.microclusters[index + 1]


    def updateSingleReliability(self,keys,currenTime,lmda,wt):

            #currentImpt=self.microclusters[keys+1 ][7]
            #previuusTime=self.microclusters[keys+1 ][6]
            #self.microclusters[keys+1 ][7] = currentImpt*(2**(-lmda*(currenTime-previuusTime)))
            self.microclusters[keys + 1][7] = self.microclusters[keys + 1][7] + 1
            self.microclusters[keys + 1][6] = currenTime
            return self


    def updateReliability(self,currenTime,lmda,wt):

        for keys in self.getMicrocluster().copy():

            currentImpt=self.microclusters[keys ][7]
            previuusTime=self.microclusters[keys ][6]
            self.microclusters[keys ][7] = currentImpt*(2**(-lmda*(currenTime-previuusTime)))


        for keys in self.getMicrocluster().copy():
            if self.microclusters[keys][7]<wt:
                   self.microclusters.pop(keys)

        #reshuffle microclsuters keys
        new_instance_cluster={}
        for index,keys in enumerate(self.getMicrocluster().copy()):
            new_instance_cluster[index+1]=self.microclusters[keys]

        self.microclusters=new_instance_cluster

        return self

    def globalUpdateReliability(self,currenTime,lmda,wt):

        for keys in self.getMicrocluster().copy():

            currentImpt=self.microclusters[keys ][7]
            previuusTime=self.microclusters[keys ][6]
            self.microclusters[keys ][7] = currentImpt*(2**(-lmda*(currenTime-previuusTime)))

        for keys in self.getMicrocluster().copy():
            if self.microclusters[keys][7] < wt:
                self.microclusters.pop(keys)

        # reshuffle microclsuters keys
        new_instance_cluster = {}
        for index, keys in enumerate(self.getMicrocluster().copy()):
            new_instance_cluster[index + 1] = self.microclusters[keys]

        self.microclusters = new_instance_cluster

        return self

    def updateMcInfo(self,data,clus_index,ctime):

        data_t = data[0:self.zero_index_features]
        class_data = int(data[self.zero_index_features])

        mc= np.asarray(self.getSingleMC(clus_index))


        LS =np.add(mc[0] , data_t)
        SS = np.add(mc[1],np.square(data_t))
        N_pt = mc[8]+1

        warnings.filterwarnings('ignore')
        mc_radius = np.sqrt(np.sum(SS / N_pt) - np.sum(np.square((LS /N_pt))))
        mc_time = ctime
        mc_cnter=LS/N_pt

        self.microclusters[clus_index+1][0] =LS
        self.microclusters[clus_index + 1][1] = SS
        self.microclusters[clus_index + 1][8] =N_pt
        self.microclusters[clus_index + 1][5] = mc_cnter
        self.microclusters[clus_index + 1][2] = mc_radius
        self.microclusters[clus_index + 1][6] = mc_time
        self.microclusters[clus_index + 1][9] = mc[9]

        if mc[4] == 0:
            self.microclusters[clus_index + 1][3] = class_data
            self.microclusters[clus_index + 1][4] = 1

        return self


    def insertClientMC(self,clusters):

        for clus_d in clusters:

          no_Instances = self.getClusInstances() + 1
          self.microclusters[no_Instances] = list(clus_d)

        return self


    def emptyMicrocluster(self):
         self.microclusters={}


    def deleteMC(self,key):


        self.microclusters.pop(key+1)

        # reshuffle microclsuters keys
        new_instance_cluster = {}
        for index, keys in enumerate(self.getMicrocluster().copy()):
            new_instance_cluster[index + 1] = self.microclusters[keys]

        self.microclusters = new_instance_cluster

        return self



    def refreshMCs(self):

        # reshuffle microclsuters keys
        new_instance_cluster = {}
        for index, keys in enumerate(self.getMicrocluster().copy()):
            new_instance_cluster[index + 1] = self.microclusters[keys]

        self.microclusters = new_instance_cluster

        return self