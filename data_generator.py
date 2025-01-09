import numpy as np

class input_data(object):

	def __init__(self):



		R_1_2 = np.load(r"D:\Data_deal\data_dealing\data\gene2miRNA.npy")
		R_1_3 = np.load(r"D:\Data_deal\data_dealing\data\gene_pathway.npy")
		R_1_4 = np.load(r"D:\Data_deal\data_dealing\data\gene2TO.npy")
		R_2_2 = np.load(r"D:\Data_deal\data_dealing\data\miRNA_miRNA.npy")
		R_2_4 = np.load(r"D:\Data_deal\data_dealing\data\miRNA_TO.npy")
		R_4_4 = np.load(r"D:\Data_deal\data_dealing\data\TO_TO.npy")

		X_1_1 = np.load(r"D:\Data_deal\data_dealing\data\gene_sep_per.npy")
		X_1_2 = np.load(r"D:\Data_deal\data_dealing\data\gene_go_pca.npy")
		# 表达数据
		X_1_3 = np.load(r"D:\Data_deal\data_dealing\data\gene_pca_expression.npy")
		X_2_1 = np.load(r"D:\Data_deal\data_dealing\data\mirna_pca_expression.npy")

		X_4_1 = np.load(r"D:\Data_deal\data_dealing\data\TO_feature_embedding.npy")
		# TO 文本语义相似性-描述
		X_4_2 = np.load(r"D:\Data_deal\pythonProject\TO_all_def_text_similarity.npy")

		# TO 文本语义相似度-名称
		# X_4_2 = np.load(r"D:\Data_deal\pythonProject\TO_text_similarity.npy")
		# X_4_2 = np.load(r"D:\Data_deal\pythonProject\denoised_similarity_matrix_pytorch.npy")

		# TO 语义相似度
		X_4_3 = np.load(r"D:\Data_deal\pythonProject\yuyi_denoised_similarity_matrix_pytorch.npy")

		self.R_2_2 = R_2_2  # miRNA序列相似度
		self.R_4_4 = R_4_4  # TO层次结构
		self.R_1_2 = R_1_2  # gene-miRNA
		self.R_1_3 = R_1_3  # gene-pathway
		self.R_1_4 = R_1_4  # gene-TO
		self.R_2_4 = R_2_4  # miRNA-TO

		# 补充
		X_1_3 = np.clip(X_1_3, a_min=0, a_max=None)
		X_1_3 = np.log2(X_1_3 + 1)
		# 补充
		X_2_1 = np.clip(X_2_1, a_min=0, a_max=None)
		X_2_1 = np.log2(X_2_1 + 1)

		X_1_1 = (X_1_1 - np.min(X_1_1)) / (np.max(X_1_1) - np.min(X_1_1))  # gene  序列特征 pca
		X_1_2 = (X_1_2 - np.min(X_1_2)) / (np.max(X_1_2) - np.min(X_1_2))  # gene  GO
		X_1_3 = (X_1_3 - np.min(X_1_3)) / (np.max(X_1_3) - np.min(X_1_3))
		X_2_1 = (X_2_1 - np.min(X_2_1)) / (np.max(X_2_1) - np.min(X_2_1))
		X_4_1 = (X_4_1 - np.min(X_4_1)) / (np.max(X_4_1) - np.min(X_4_1))  # TO层次结构 词表征学习
		# TO
		X_4_2 = (X_4_2 - np.min(X_4_2)) / (np.max(X_4_2) - np.min(X_4_2))

		X_4_3 = (X_4_3 - np.min(X_4_3)) / (np.max(X_4_3) - np.min(X_4_3))

		self.X_1_1 = X_1_1
		self.X_1_2 = X_1_2
		self.X_1_3 = X_1_3
		self.X_2_1 = X_2_1
		self.X_4_1 = X_4_1
		# TO
		self.X_4_2 = X_4_2
		self.X_4_3 = X_4_3




















