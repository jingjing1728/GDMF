import torch
import torch.optim as optim
import re
from sklearn.decomposition import PCA
from torch_geometric.data import HeteroData

import data_generator
import tools
import torch.utils.data as Data

from args import read_args
import copy
from torch.autograd import Variable
import numpy as np
import random

torch.set_num_threads(2)
import os


tools.same_seed(1)



class model_class(object):

    def __init__(self, args):

        super(model_class, self).__init__()

        self.args = args

        input_data = data_generator.input_data()

        gene_feature_dim = input_data.X_1_1.shape[1] + input_data.X_1_2.shape[1] + input_data.X_1_3.shape[1]
        mirna_feature_dim = input_data.X_2_1.shape[1]
        phenotype_feature_dim = input_data.X_4_1.shape[1] + input_data.X_4_3.shape[1]

        self.gene_feature_dim = gene_feature_dim
        self.mirna_feature_dim = mirna_feature_dim
        self.phenotype_feature_dim = phenotype_feature_dim

        print('Data loading successfully!')

        self.R_X_dict = {
            'R_2_2': input_data.R_2_2,
            'R_4_4': input_data.R_4_4,
            'R_1_2': input_data.R_1_2,
            'R_1_3': input_data.R_1_3,
            'R_1_4': input_data.R_1_4,
            'R_2_4': input_data.R_2_4,

            'X_1_1': input_data.X_1_1,
            'X_1_2': input_data.X_1_2,
            'X_1_3': input_data.X_1_3,
            'X_2_1': input_data.X_2_1,
            'X_4_1': input_data.X_4_1,
            'X_4_2': input_data.X_4_2,
            'X_4_3': input_data.X_4_3

        }

        self.gene_num = self.R_X_dict['R_1_4'].shape[0]

        # obtaining layer id
        lay_id = set()
        for k in self.R_X_dict:
            if (re.split("_", k)[0] == 'R'):
                layer1 = re.split("_", k)[1]
                layer2 = re.split("_", k)[2]
                if (layer1 == layer2):
                    lay_id.add(layer1)
                else:
                    lay_id.add(layer1)
                    lay_id.add(layer2)

        G_input_dict = {}
        G_ini_dict = {}
        GiUt_ini_dict = {}
        for i in lay_id:
            for k in self.R_X_dict:
                if(re.split("_", k)[0] == 'R'):
                    layer1 = re.split("_", k)[1]
                    layer2 = re.split("_", k)[2]
                    if (i == layer1):
                        if (('G' + i) not in G_input_dict.keys()):
                            G_input_dict['G' + i] = self.R_X_dict[k]
                        else:
                            G_input_dict['G' + i] = np.hstack((G_input_dict['G' + i], self.R_X_dict[k]))
                    if ((i != layer1) & (i == layer2)):
                        if (('G' + i) not in G_input_dict.keys()):
                            G_input_dict['G' + i] = self.R_X_dict[k].T
                        else:
                            G_input_dict['G' + i] = np.hstack((G_input_dict['G' + i], self.R_X_dict[k].T))
                if (re.split("_", k)[0] == 'X'):
                    layer = re.split("_", k)[1]
                    layer_t = re.split("_", k)[2]    # GiUt
                    if (i == layer):
                        if (('G' + i) not in G_input_dict.keys()):
                            G_input_dict['G' + i] = self.R_X_dict[k]
                        else:
                            G_input_dict['G' + i] = np.hstack((G_input_dict['G' + i], self.R_X_dict[k]))
                        GiUt_ini_dict['G' + i + 'U' + layer_t] = np.ones((self.R_X_dict[k].shape[1], self.args.ini_d))
            G_ini_dict['G' + i] = self.get_G_ini(G_input_dict['G' + i], self.args.ini_d)

        for i in lay_id:
            for k in self.R_X_dict:
                if (re.split("_", k)[0] == 'X'):
                    layer = re.split("_", k)[1]
                    layer_t = re.split("_", k)[2]    # GiUt
                    if (i == layer):
                        # GiUt_ini_dict['G' + i + 'U' + layer_t] = np.ones((self.R_X_dict[k].shape[1], self.args.ini_d))
                        GiUt_ini_dict['G' + i + 'U' + layer_t] = np.matmul(np.linalg.inv(G_ini_dict['G' + i][:self.args.ini_d]), self.R_X_dict[k][:self.args.ini_d]).T

        R_1_1 = np.load(r"D:\Data_deal\data_dealing\data\ppi.npy")
        R_1_2 = np.load(r"D:\Data_deal\data_dealing\data\gene2miRNA.npy")
        R_1_3 = np.load(r"D:\Data_deal\data_dealing\data\gene_pathway.npy")
        R_1_4 = np.load(r"D:\Data_deal\data_dealing\data\gene2TO.npy")

        R_2_2 = np.load(r"D:\Data_deal\data_dealing\data\miRNA_miRNA.npy")
        R_2_4 = np.load(r"D:\Data_deal\data_dealing\data\miRNA_TO.npy")

        R_4_4 = np.load(r"D:\Data_deal\data_dealing\data\TO_TO.npy")

        X_1_1 = np.load(r"D:\Data_deal\data_dealing\data\gene_sep_per.npy")
        X_1_2 = np.load(r"D:\Data_deal\data_dealing\data\gene_go_pca.npy")
        X_1_3 = np.load(r"D:\Data_deal\data_dealing\data\gene_pca_expression.npy")

        X_2_1 = np.load(r"D:\Data_deal\data_dealing\data\mirna_pca_expression.npy")

        X_4_1 = np.load(r"D:\Data_deal\data_dealing\data\TO_feature_embedding.npy")
        X_4_2 = np.load(r"D:\Data_deal\pythonProject\TO_all_def_text_similarity.npy")
        X_4_3 = np.load(r"D:\Data_deal\pythonProject\yuyi_denoised_similarity_matrix_pytorch.npy")
        
        R_1_1_tensor = torch.from_numpy(R_1_1)
        R_2_2_tensor = torch.from_numpy(R_2_2)
        R_4_4_tensor = torch.from_numpy(R_4_4)
        R_1_2_tensor = torch.from_numpy(R_1_2)
        R_1_4_tensor = torch.from_numpy(R_1_4)
        X_1_1_tensor = torch.from_numpy(X_1_1)
        X_1_2_tensor = torch.from_numpy(X_1_2)
        X_1_3_tensor = torch.from_numpy(X_1_3)
        X_2_1_tensor = torch.from_numpy(X_2_1)
        X_4_2_tensor = torch.from_numpy(X_4_2)
        X_4_1_tensor = torch.from_numpy(X_4_1)
        X_4_3_tensor = torch.from_numpy(X_4_3)

        gene_features = torch.cat((X_1_1_tensor,X_1_2_tensor,X_1_3_tensor),dim=1)
        # print(gene_features.shape)
        miRNA_features = X_2_1_tensor
        # print(miRNA_features.shape)
        TO_features = torch.cat((X_4_1_tensor,X_4_3_tensor),dim=1)
        # print(TO_features.shape)

        data = HeteroData()

        data['gene'].x = gene_features
        data['miRNA'].x = miRNA_features
        data['TO'].x = TO_features

        row_indices, col_indices = R_1_2.nonzero()

        edge_index_g2m = torch.tensor([row_indices, col_indices], dtype=torch.long)

        data['gene', 'regulated_by', 'mirna'].edge_index = edge_index_g2m

        row_indices, col_indices = R_1_3.nonzero()

        edge_index_g2p = torch.tensor([row_indices, col_indices], dtype=torch.long)

        data['gene', 'regulated_by', 'pathway'].edge_index = edge_index_g2p

        row_indices, col_indices = R_1_4.nonzero()

        edge_index_g2t = torch.tensor([row_indices, col_indices], dtype=torch.long)

        data['gene', 'regulated_by', 'TO'].edge_index = edge_index_g2t

        row_indices, col_indices = R_2_4.nonzero()

        edge_index_M2t = torch.tensor([row_indices, col_indices], dtype=torch.long)
        data['mirna', 'regulated_by', 'TO'].edge_index = edge_index_M2t

        edge_index_miRNA_miRNA = torch.tensor(np.nonzero(R_2_2), dtype=torch.long)
        edge_index_TO_TO = torch.tensor(np.nonzero(R_4_4), dtype=torch.long)

        edge_index_gene_gene = torch.tensor(np.nonzero(R_2_2),dtype=torch.long)
        data['gene', 'similar_to', 'gene'].edge_index = edge_index_gene_gene

        data['mirna', 'similar_to', 'mirna'].edge_index = edge_index_miRNA_miRNA
        data['TO', 'is_a', 'TO'].edge_index = edge_index_TO_TO

        edge_index_TO_TO_sem_sim = torch.tensor(np.nonzero(X_4_3), dtype=torch.long)
        data['TO', 'similar_to', 'TO'].edge_index = edge_index_TO_TO_sem_sim

        self.gat_model = tools.HeteroGAT(self.gene_feature_dim, self.mirna_feature_dim, self.phenotype_feature_dim)
        self.gat_model.init_weights()
        self.G_ini_dict = G_ini_dict
        self.GiUt_ini_dict = GiUt_ini_dict


        for k in self.R_X_dict:
            self.R_X_dict[k] = torch.from_numpy(np.array(self.R_X_dict[k])).float()

        for k in self.G_ini_dict:
            self.G_ini_dict[k] = torch.from_numpy(np.array(self.G_ini_dict[k])).float()

        for k in self.GiUt_ini_dict:
            self.GiUt_ini_dict[k] = torch.from_numpy(np.array(self.GiUt_ini_dict[k])).float()

        if torch.cuda.is_available():
            for k in self.R_X_dict:
                self.R_X_dict[k] = self.R_X_dict[k].cuda()
            for k in self.G_ini_dict:
                self.G_ini_dict[k] = self.G_ini_dict[k].cuda()
            for k in self.GiUt_ini_dict:
                self.GiUt_ini_dict[k] = self.GiUt_ini_dict[k].cuda()

        self.model = tools.WDGPA(args)
        self.gat_model = tools.HeteroGAT(self.gene_feature_dim, self.mirna_feature_dim, self.phenotype_feature_dim)

        if torch.cuda.is_available():
            self.model.cuda()

        self.parameters = self.model.parameters()
        self.parameters = list(self.model.parameters()) + list(self.gat_model.parameters())

        self.model.init_weights()

        self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay=0)


        x = torch.linspace(0, self.gene_num-1, self.gene_num)   # this is x data (torch tensor)

        torch_dataset = Data.TensorDataset(x)
        self.loader = Data.DataLoader(  # 使训练变成一小批的
            dataset=torch_dataset,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # mini batch size
            shuffle=True,  # random shuffle for training
            num_workers=2,  # subprocesses for loading data
        )

    def select_batch(self, dict, type, batch):
        copy_dict = copy.deepcopy(dict)
        if(type == 'R_X'):
            for k in copy_dict:
                if (re.split("_", k)[0] == 'R'):
                    layer1 = re.split("_", k)[1]
                    if (layer1 == '1'):
                        copy_dict[k] = copy_dict[k][batch]
                if (re.split("_", k)[0] == 'X'):
                    layer = re.split("_", k)[1]
                    if (layer == '1'):
                        copy_dict[k] = copy_dict[k][batch]
        elif(type == 'G_ini'):
            copy_dict['G1'] = copy_dict['G1'][batch]
        return copy_dict


    def get_G_ini(self, G_input, dim):
        pca = PCA(n_components=dim)
        G_ini = pca.fit_transform(G_input)
        return G_ini



    def model_train(self):

        for epoch in range(self.args.train_iter_n):

            print(f'Training at epoch: {epoch + 1}/{self.args.train_iter_n}')  # 打印当前epoch

            loss_sum = 0

            loss_R14_sum = 0

            for step, batch_x in enumerate(self.loader):  # for each training step

                # print('epoch: ', epoch, '       ', 'step: ', step)

                loss, loss_R14, Y_1_4, sum_w, wr_list = self.model(self.select_batch(self.R_X_dict, 'R_X', batch_x[0].numpy()),self.select_batch(self.G_ini_dict, 'G_ini', batch_x[0].numpy()),self.GiUt_ini_dict, epoch)
                loss_sum += loss
                loss_R14_sum += loss_R14


                self.optim.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                self.optim.step()  # apply gradients




            print(sum_w, wr_list.T)
            print(loss_sum, loss_R14_sum)
            print('--------------------**-------------------**--------------------**---------------------')









    def model_case_study(self):
        loss, loss_R14, Y_1_4, sum_w, wr_list = self.model(self.R_X_dict,self.G_ini_dict,self.GiUt_ini_dict, 0)
        if torch.cuda.is_available():
            tools.get_evaluation_metrics(Y_1_4.cpu().detach().numpy(), self.R_X_dict['R_1_4'].cpu().detach().numpy(), wr_list.cpu().detach().numpy())
            np.save('./result/case_study.npy', Y_1_4.cpu().detach().numpy())
        else:
            tools.get_evaluation_metrics(Y_1_4.detach().numpy(), self.R_X_dict['R_1_4'].detach().numpy(), wr_list.detach().numpy() )
            np.save('./result/case_study.npy', Y_1_4.detach().numpy())



if __name__ == '__main__':
    args = read_args()

    # model
    print("------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>model_class<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("------------------------------------------------------------------------")
    model_object = model_class(args)


    print("------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>model_train<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("------------------------------------------------------------------------")
    model_object.model_train()


    print("------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>model_evaluate<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("------------------------------------------------------------------------")
    model_object.model_case_study()

