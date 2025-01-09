import numpy as np

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
# TO 文本语义相似性-描述
X_4_2 = np.load(r"D:\Data_deal\pythonProject\TO_all_def_text_similarity.npy")
# TO 文本语义相似度-名称
# X_4_2 = np.load(r"D:\Data_deal\pythonProject\TO_text_similarity.npy")
# X_4_2 = np.load(r"D:\Data_deal\pythonProject\denoised_similarity_matrix_pytorch.npy")
X_4_3 = np.load(r"D:\Data_deal\pythonProject\yuyi_denoised_similarity_matrix_pytorch.npy")

# TO 语义相似度
# X_4_3 = np.load(r"D:\Data_deal\pythonProject\yuyi_denoised_similarity_matrix_pytorch.npy")
print("R_1_1:基因12098相似性（BLAST/PPI）TAN")
print("R_1_2:基因12098-miRNA447 TAN")
print("R_1_3:基因12098-pathway368 sql")
print("R_1_4:基因12098-TO155")

print("R_2_2:miRNA相似性 TAN(Liang)")
print("R_2_4:miRNA-TO 矩阵相乘取阈值")

print("R_4_4:TO-TO 层次关联 i到j 添加一条边 (父 -> 子)")

print("X_1_1:基因序列 4-mer")
print("X_1_2:GO富集）")
print("X_1_3:基因表达")

print("X_2_1:miRNA表达")

print("X_4_1:TO嵌入 DeepWalk")
print("X_4_2:文本")
print("X_4_3:语义 降噪")
