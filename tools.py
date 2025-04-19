import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import re

from torch_geometric.nn import GATConv

import torch.nn.init as init


def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
same_seed(1)


def liner(x):
    return x


class HeteroGAT(torch.nn.Module):
    def __init__(self, gene_feature_dim, mirna_feature_dim, phenotype_feature_dim):
        super(HeteroGAT, self).__init__()
        self.gene_conv1 = GATConv(gene_feature_dim, 256, heads=4, dropout=0.2)
        # 737
        self.gene_conv2 = GATConv(256 * 4, 256, dropout=0.2)
        self.gene_pathway_conv = GATConv(256, 256, dropout=0.2)
        self.gene_miRNA_conv = GATConv(256, 256, dropout=0.2)  
        self.gene_TO_conv = GATConv(256, 256, dropout=0.2)
        # self.conv_mirna_features = GATConv(mirna_feature_dim, 256, dropout=0.2)
        # self.conv_phenotype_features = GATConv(phenotype_feature_dim, 256, dropout=0.2)

        self.miRNA_conv1 = GATConv(mirna_feature_dim, 128, heads=8, dropout=0.2)
        self.miRNA_conv2 = GATConv(128 * 8, 128, dropout=0.2)
        self.miRNA_gene_conv = GATConv(128, 128, dropout=0.2) 
        self.miRNA_TO_conv = GATConv(128, 128, dropout=0.2)
        # self.conv_mirna_features = GATConv(mirna_feature_dim, 128, dropout=0.2)
        # self.conv_phenotype_features = GATConv(phenotype_feature_dim, 128, dropout=0.2)
        #
        self.TO_conv1 = GATConv(phenotype_feature_dim, 256, heads=4, dropout=0.2)
        # 283
        self.TO_conv2 = GATConv(256 * 4, 128, dropout=0.2)
        self.TO_conv3 = GATConv(128, 128, dropout=0.2)
        self.TO_gene_conv = GATConv(128, 128, dropout=0.2)  
        self.TO_miRNA_conv = GATConv(128, 128, dropout=0.2)
        # self.conv_mirna_features = GATConv(mirna_feature_dim, 256, dropout=0.2)
        # self.conv_phenotype_features = GATConv(phenotype_feature_dim, 256, dropout=0.2)

        self.fc = nn.Linear(256, 128) 


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')  # 使用ReLU的He初始化
                m.bias.data.fill_(0)

            elif isinstance(m, GATConv):
                if hasattr(m, 'weight'):
                    nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if hasattr(m, 'att'):
                    nn.init.kaiming_uniform_(m.att.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x, edge_index_dict):
        gene_x = self.gene_conv1(x['gene'], edge_index_dict[('gene', 'similar_to', 'gene')])
        gene_x = F.elu(gene_x)  
        gene_x = F.dropout(gene_x, p=0.5, training=self.training)
        gene_x = self.gene_conv2(gene_x, edge_index_dict[('gene', 'similar_to', 'gene')])
        gene_x = self.gene_pathway_conv(gene_x, edge_index_dict[('gene', 'regulated_by', 'pathway')])
        gene_x = F.elu(gene_x)
        gene_x = self.gene_miRNA_conv(gene_x, edge_index_dict[('gene', 'regulated_by', 'mirna')])  
        gene_x = F.elu(gene_x)
        gene_x = self.gene_TO_conv(gene_x,
                                   edge_index_dict[('gene', 'regulated_by', 'TO')])  
        gene_x = F.elu(gene_x)
        gene_x = self.fc(gene_x)

        miRNA_x = self.miRNA_conv1(x['miRNA'], edge_index_dict[('mirna', 'similar_to', 'mirna')])
        miRNA_x = F.elu(miRNA_x)
        miRNA_x = F.dropout(miRNA_x, p=0.5, training=self.training)
        miRNA_x = self.miRNA_conv2(miRNA_x, edge_index_dict[('mirna', 'similar_to', 'mirna')])

        miRNA_x = self.miRNA_gene_conv(miRNA_x, edge_index_dict[('gene', 'regulated_by', 'mirna')]) 
        miRNA_x = F.elu(miRNA_x)

        miRNA_x = self.miRNA_TO_conv(miRNA_x,
                                     edge_index_dict[('mirna', 'regulated_by', 'TO')]) 
        # miRNA_x = self.fc2(miRNA_x)

        TO_x = self.TO_conv1(x['TO'], edge_index_dict[('TO', 'is_a', 'TO')])
        TO_x = F.elu(TO_x)
        TO_x = F.dropout(TO_x, p=0.5, training=self.training)
        TO_x = self.TO_conv2(TO_x, edge_index_dict[('TO', 'is_a', 'TO')])

        TO_x = self.TO_gene_conv(TO_x, edge_index_dict[('gene', 'regulated_by', 'TO')])  
        TO_x = F.elu(TO_x)

        TO_x = self.TO_miRNA_conv(TO_x,
                                  edge_index_dict[('mirna', 'regulated_by', 'TO')])
        TO_x = F.elu(TO_x)
        # TO_x = self.fc2(TO_x)

        return gene_x,miRNA_x,TO_x
 
class WDGPA(nn.Module):
    def __init__(self, args, gat_model = None):
        super(WDGPA, self).__init__()
        self.gat_model = gat_model

        # self.act = liner
        # self.act = nn.ReLU()
        self.act = nn.LeakyReLU()

        self.Layers = args.Layers
        self.ini_d = args.ini_d
        self.train_iter_n = args.train_iter_n
        self.alpha = args.alpha
        self.beta = args.beta

        self.Layer_1_1_run = nn.Linear(self.ini_d, self.Layers['Layer_1'][0])
        self.Layer_1_2_run = nn.Linear(self.Layers['Layer_1'][0], self.Layers['Layer_1'][1])
        self.Layer_2_1_run = nn.Linear(self.ini_d, self.Layers['Layer_2'][0])
        self.Layer_2_2_run = nn.Linear(self.Layers['Layer_2'][0], self.Layers['Layer_2'][1])
        self.Layer_3_1_run = nn.Linear(self.ini_d, self.Layers['Layer_3'][0])
        self.Layer_3_2_run = nn.Linear(self.Layers['Layer_3'][0], self.Layers['Layer_3'][1])
        self.Layer_4_1_run = nn.Linear(self.ini_d, self.Layers['Layer_4'][0])
        self.Layer_4_2_run = nn.Linear(self.Layers['Layer_4'][0], self.Layers['Layer_4'][1])

        self.wr_list = nn.Parameter(torch.FloatTensor(torch.ones(6, 1)), requires_grad=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.1)


    def cos_sim(self, a, b):
        feature1 = F.normalize(a)
        feature2 = F.normalize(b)
        distance1 = feature1.mm(feature2.t())
        return distance1

    def forward(self,R_X_dict, G_ini_dict, GiUt_ini_dict, iter_i):

        G_output_dict = {}
        U_output_dict = {}
        Y_dict = {}

        if self.gat_model:
            gene_embeddings, miRNA_embeddings, TO_embeddings = self.gat_model(
                self.data.x_dict, self.data.edge_index_dict
            )

            G_ini_dict['G1'] = gene_embeddings  
            G_ini_dict['G2'] = miRNA_embeddings
            G_ini_dict['G4'] = TO_embeddings  

        G_output_dict['G1'] = self.Layer_1_1_run(G_ini_dict['G1'])
        G_output_dict['G1'] = self.act(G_output_dict['G1'])
        G_output_dict['G1'] = self.Layer_1_2_run(G_output_dict['G1'])
        G_output_dict['G1'] = self.act(G_output_dict['G1'])

        G_output_dict['G2'] = self.Layer_2_1_run(G_ini_dict['G2'])
        G_output_dict['G2'] = self.act(G_output_dict['G2'])
        G_output_dict['G2'] = self.Layer_2_2_run(G_output_dict['G2'])
        G_output_dict['G2'] = self.act(G_output_dict['G2'])

        G_output_dict['G3'] = self.Layer_3_1_run(G_ini_dict['G3'])
        G_output_dict['G3'] = self.act(G_output_dict['G3'])
        G_output_dict['G3'] = self.Layer_3_2_run(G_output_dict['G3'])
        G_output_dict['G3'] = self.act(G_output_dict['G3'])

        G_output_dict['G4'] = self.Layer_4_1_run(G_ini_dict['G4'])
        G_output_dict['G4'] = self.act(G_output_dict['G4'])
        G_output_dict['G4'] = self.Layer_4_2_run(G_output_dict['G4'])
        G_output_dict['G4'] = self.act(G_output_dict['G4'])

        sm = nn.Softmax(dim=0)

        wr_list = sm(self.wr_list)

        for k in R_X_dict:
            if (re.split("_", k)[0] == 'R'):
                layer1 = re.split("_", k)[1]
                layer2 = re.split("_", k)[2]
                if (self.act == liner):
                    Y_dict['Y_' + layer1 + '_' + layer2] = torch.mm(G_output_dict['G' + layer1],
                                                                    G_output_dict['G' + layer2].T)
                else:
                    Y_dict['Y_' + layer1 + '_' + layer2] = torch.mm(G_output_dict['G' + layer1],
                                                                    G_output_dict['G' + layer2].T)

        loss = 0
        i = 0
        j = 0
        for k in R_X_dict:
            if (re.split('_', k)[0] == 'R'):
                layer1 = re.split("_", k)[1]
                layer2 = re.split("_", k)[2]
                loss += sum(sum((R_X_dict[k] - Y_dict['Y_' + layer1 + '_' + layer2]) ** 2))
                i += 1

        loss_R14 = sum(sum((R_X_dict['R_1_4'] - Y_dict['Y_1_4']) ** 2))

        sum_w = 0

        return loss + sum_w, loss_R14, Y_dict['Y_1_4'], sum_w, wr_list


def get_evaluation_metrics(prediction_score, Y, wr_list):
    IC = np.zeros((155, 1))
    # Y = (np.abs(Y) + Y) / 2
    m = Y.shape[0]
    n = Y.shape[1]
    prediction_score_col = prediction_score.reshape(-1)
    Indices = np.arange(len(prediction_score_col))

    np.random.seed(1)

    np.random.shuffle(Indices)
    X = Indices
    thresholds_num = 2000
    X = X[:thresholds_num]
    thresholds = prediction_score_col[X]
    thresholds = np.sort(thresholds)[::-1]  # descend

    tpr = np.zeros((1, thresholds_num)).reshape(-1)
    fpr = np.zeros((1, thresholds_num)).reshape(-1)
    precision = np.zeros((1, thresholds_num)).reshape(-1)
    recall = np.zeros((1, thresholds_num)).reshape(-1)

    Fmax = 0
    Smin = 1000000

    for qq in range(thresholds_num):
        prediction = (prediction_score >= thresholds[qq]) + 0
        TP = np.sum(Y * prediction)
        pre_pos = np.sum(prediction)
        FP = pre_pos - TP
        pre_neg = m * n - pre_pos
        TN = np.sum((prediction + Y == 0) + 0)
        FN = pre_neg - TN

        RU = (prediction - Y == 1) + 0
        ru = np.sum(RU @ IC) / m

        MI = (Y - prediction == 1) + 0
        mi = np.sum(MI @ IC) / m

        s = np.sqrt(ru ** 2 + mi ** 2)
        if (s < Smin):
            Smin = s

        if (TP + FN != 0):
            tpr[qq] = TP / (TP + FN)
            recall[qq] = TP / (TP + FN)
        if (TN + FP != 0):
            fpr[qq] = FP / (TN + FP)
        if (TP + FP != 0):
            precision[qq] = TP / (TP + FP)

        f = 2 * precision[qq] * recall[qq] / (precision[qq] + recall[qq] + 1e-20)
        if (f > Fmax):
            Fmax = f

    plt.figure()
    plt.plot(fpr, tpr)

    plt.figure()
    plt.plot(recall, precision)

    plt.show()

    auroc = np.zeros((1, thresholds_num - 1)).reshape(-1)
    auprc = np.zeros((1, thresholds_num - 1)).reshape(-1)
    for k in range(thresholds_num - 1):
        auroc[k] = (fpr[k + 1] - fpr[k]) * (tpr[k + 1] + tpr[k]) / 2
        auprc[k] = (recall[k + 1] - recall[k]) * (precision[k + 1] + precision[k]) / 2

    AUROC = np.sum(auroc)
    AUPRC = np.sum(auprc)

    print('AUROC: ', AUROC, '  AUPRC: ', AUPRC, '  Fmax: ', Fmax, '  Smin: ', Smin)

    res_dict = {
        'prediction_score': prediction_score,
        'AUROC': AUROC,
        'AUPRC': AUPRC,
        'Fmax': Fmax,
        'Smin': Smin,
        'wr_list': wr_list,
    }

    return res_dict






