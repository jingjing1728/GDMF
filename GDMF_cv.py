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
os.environ["CUDA_VISIBLE_DEVICES"] = '2'



def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
same_seed(1)

class cross_validation(object):

    def __init__(self, args):

        super(cross_validation, self).__init__()

        self.fold = 5

        self.args = args

        input_data = data_generator.input_data()

        # 计算特征维度
        gene_feature_dim = input_data.X_1_1.shape[1] + input_data.X_1_2.shape[1] + input_data.X_1_3.shape[1]
        mirna_feature_dim = input_data.X_2_1.shape[1]
        phenotype_feature_dim = input_data.X_4_1.shape[1] + input_data.X_4_2.shape[1]

        # 确保特征维度被正确设置
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
            'X_4_2': input_data.X_4_2
        }

        self.gene_num = self.R_X_dict['R_1_4'].shape[0]

        # 得到层号
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

        # 初始化Gi    GiUt
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



        # 假设所有数据都已经加载到NumPy数组中
        R_1_1 = np.load(r"D:\Data_deal\data_dealing\data\ppi.npy")
        R_2_2 = np.load(r"D:\Data_deal\data_dealing\data\miRNA_miRNA.npy")
        print(R_2_2.size - np.count_nonzero(R_2_2))
        R_4_4 = np.load(r"D:\Data_deal\data_dealing\data\TO_TO.npy")
        print(R_4_4.size - np.count_nonzero(R_4_4))
        R_1_2 = np.load(r"D:\Data_deal\data_dealing\data\gene2miRNA.npy")
        print(np.sum(R_1_2))
        R_1_3 = np.load(r"D:\Data_deal\data_dealing\data\gene_pathway.npy")
        print(np.sum(R_1_3))
        R_1_4 = np.load(r"D:\Data_deal\data_dealing\data\gene2TO.npy")
        print(np.sum(R_1_4))
        R_2_4 = np.load(r"D:\Data_deal\data_dealing\data\miRNA_TO.npy")
        print(np.sum(R_2_4))
        X_1_1 = np.load(r"D:\Data_deal\data_dealing\data\gene_sep_per.npy")
        print(X_1_1.shape)
        X_1_2 = np.load(r"D:\Data_deal\data_dealing\data\gene_go_pca.npy")
        X_1_3 = np.load(r"D:\Data_deal\data_dealing\data\gene_pca_expression.npy")
        X_2_1 = np.load(r"D:\Data_deal\data_dealing\data\mirna_pca_expression.npy")
        X_4_2 = np.load(r"D:\Data_deal\pythonProject\denoised_similarity_matrix_pytorch.npy")
        X_4_1 = np.load(r"D:\Data_deal\data_dealing\data\TO_feature_embedding.npy")
        X_4_3 = np.load(r"D:\Data_deal\pythonProject\yuyi_denoised_similarity_matrix_pytorch.npy")

        # 将NumPy数组转换为PyTorch张量
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

        gene_features = torch.cat((X_1_1_tensor, X_1_2_tensor, X_1_3_tensor), dim=1)
        print(gene_features.shape)
        miRNA_features = X_2_1_tensor
        print(miRNA_features.shape)
        TO_features = torch.cat((X_4_1_tensor, X_4_2_tensor, X_4_3_tensor), dim=1)
        print(TO_features.shape)

        # 创建异构图数据对象
        data = HeteroData()

        data['gene'].x = gene_features
        data['miRNA'].x = miRNA_features
        data['TO'].x = TO_features

        # 将 R_1_2 转换为边索引
        row_indices, col_indices = R_1_2.nonzero()

        # 将 NumPy 数组转换为 PyTorch 张量
        edge_index_g2m = torch.tensor([row_indices, col_indices], dtype=torch.long)

        # 添加边索引到数据对象
        data['gene', 'regulated_by', 'mirna'].edge_index = edge_index_g2m

        # 将 R_1_3 转换为边索引
        row_indices, col_indices = R_1_3.nonzero()

        # 将 NumPy 数组转换为 PyTorch 张量
        edge_index_g2p = torch.tensor([row_indices, col_indices], dtype=torch.long)

        # 添加边索引到数据对象
        data['gene', 'regulated_by', 'pathway'].edge_index = edge_index_g2p

        # 将 R_1_4 转换为边索引
        row_indices, col_indices = R_1_4.nonzero()

        # 将 NumPy 数组转换为 PyTorch 张量
        edge_index_g2t = torch.tensor([row_indices, col_indices], dtype=torch.long)

        # 添加边索引到数据对象
        data['gene', 'regulated_by', 'TO'].edge_index = edge_index_g2t

        # 将 R_2_4 转换为边索引
        row_indices, col_indices = R_2_4.nonzero()

        # 将 NumPy 数组转换为 PyTorch 张量
        edge_index_M2t = torch.tensor([row_indices, col_indices], dtype=torch.long)

        # 添加边索引到数据对象
        data['mirna', 'regulated_by', 'TO'].edge_index = edge_index_M2t

        # 假设R_2_2是miRNA的相似性矩阵，R_4_4是TO的层次结构矩阵
        # 将相似性矩阵和层次结构矩阵转换为边索引
        edge_index_miRNA_miRNA = torch.tensor(np.nonzero(R_2_2), dtype=torch.long)
        edge_index_TO_TO = torch.tensor(np.nonzero(R_4_4), dtype=torch.long)

        edge_index_gene_gene = torch.tensor(np.nonzero(R_2_2), dtype=torch.long)
        data['gene', 'similar_to', 'gene'].edge_index = edge_index_gene_gene

        # 添加边索引
        data['mirna', 'similar_to', 'mirna'].edge_index = edge_index_miRNA_miRNA
        data['TO', 'is_a', 'TO'].edge_index = edge_index_TO_TO

        edge_index_TO_TO_sem_sim = torch.tensor(np.nonzero(X_4_3), dtype=torch.long)
        data['TO', 'similar_to', 'TO'].edge_index = edge_index_TO_TO_sem_sim

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





        x = torch.linspace(0, self.gene_num-1, self.gene_num)  # this is x data (torch tensor)

        torch_dataset = Data.TensorDataset(x)
        self.loader = Data.DataLoader(  # 使训练变成一小批的
            dataset=torch_dataset,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # mini batch size
            shuffle=True,  # random shuffle for training
            num_workers=0,  # subprocesses for loading data
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


    def get_cross_validation(self):

        # 把R_X_dict中的R14分解成fold份    每份喂给model

        seed = 1
        np.random.seed(seed)

        m = self.R_X_dict['R_1_4'].shape[0]
        n = self.R_X_dict['R_1_4'].shape[1]

        all = m * n
        allIndex = np.zeros((all, 2))

        i = 0
        for col in range(n):
            for row in range(m):
                allIndex[i][1] = col
                allIndex[i][0] = row
                i += 1

        Indices = np.arange(all)
        X = copy.deepcopy(Indices)
        np.random.shuffle(X)

        HMDD = copy.deepcopy(allIndex)
        prediction_score = np.zeros((m, n))

        for cv in range(self.fold):
            print('--------Current fold: %d---------\n' % (cv+1))
            R14_temp = copy.deepcopy(self.R_X_dict['R_1_4'])
            if cv < self.fold - 1:
                B = HMDD[X[cv * int(all / self.fold):int(all / self.fold) * (cv + 1)]][:]
                for i in range(int(all / self.fold)):
                    R14_temp[int(B[i][0])][int(B[i][1])] = 0
            else:
                B = HMDD[X[cv * int(all / self.fold):all]][:]
                for i in range(all - int(all / self.fold) * (self.fold - 1)):
                    R14_temp[int(B[i][0])][int(B[i][1])] = 0

            R_X_dict = copy.deepcopy(self.R_X_dict)

            R_X_dict['R_1_4'] = R14_temp

            self.gat_model = tools.HeteroGAT(self.gene_feature_dim, self.mirna_feature_dim, self.phenotype_feature_dim)
            self.gat_model.init_weights()
            self.model = tools.WDGPA(args)

            if torch.cuda.is_available():
                self.model.cuda()

            self.model.init_weights()

            self.parameters = self.model.parameters()
            self.parameters = list(self.model.parameters()) + list(self.gat_model.parameters())

            self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay=0)


            for epoch in range(self.args.train_iter_n):

                loss_sum = 0

                loss_R14_sum = 0

                for step, batch_x in enumerate(self.loader):  # for each training step

                    print('cv: ', cv+1, '         epoch: ', epoch, '          step: ', step)

                    loss, loss_R14, Y_1_4, sum_w, wr_list, wh_list = self.model(self.select_batch(R_X_dict, 'R_X', batch_x[0].numpy()),
                                                       self.select_batch(self.G_ini_dict, 'G_ini', batch_x[0].numpy()),
                                                       self.GiUt_ini_dict, epoch)
                    loss_sum += loss
                    loss_R14_sum += loss_R14

                    self.optim.zero_grad()  # clear gradients for next train
                    loss.backward()  # backpropagation, compute gradients
                    self.optim.step()  # apply gradients
                print('cv: ', cv + 1, '         epoch: ', epoch)
                print(loss_sum, loss_R14_sum)
                print(wr_list.T, wh_list.T)

            loss, loss_R14, F, sum_w, wr_list, wh_list = self.model(R_X_dict, self.G_ini_dict, self.GiUt_ini_dict, 0)

            test_num = B.shape[0]
            for ii in range(test_num):
                prediction_score[int(B[ii][0])][int(B[ii][1])] = F[int(B[ii][0])][int(B[ii][1])]

            self.prediction_score = prediction_score
            self.wr_list = wr_list
            self.wh_list = wh_list



    def model_evaluate(self):
        if torch.cuda.is_available():
            res_dict = tools.get_evaluation_metrics(self.prediction_score, self.R_X_dict['R_1_4'].cpu().detach().numpy(), self.wr_list.cpu().detach().numpy(), self.wh_list.cpu().detach().numpy())
        else:
            res_dict = tools.get_evaluation_metrics(self.prediction_score, self.R_X_dict['R_1_4'].detach().numpy(), self.wr_list.detach().numpy(), self.wh_list.detach().numpy())
        return res_dict


if __name__ == '__main__':
    args = read_args()
    # model
    print("------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>model_class<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("------------------------------------------------------------------------")
    model_object = cross_validation(args)

    print("------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>model_train<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("------------------------------------------------------------------------")
    model_object.get_cross_validation()

    print("------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>model_evaluate<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("------------------------------------------------------------------------")
    res_dict = model_object.model_evaluate()
    np.save('./result/res_dict.npy', res_dict)




























