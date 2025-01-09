import argparse


# Maize
def read_args():

	parser = argparse.ArgumentParser()

	parser.add_argument('--train_iter_n', type=int, default=150, help = 'max number of training iteration')

	parser.add_argument('--batch_size', type=int, default=512, help = 'batch_size')

	parser.add_argument('--ini_d', type=int, default=128, help = 'initialized dimension')

	parser.add_argument('--lr', type=int, default=1e-3, help='learning rate')

	parser.add_argument('--alpha', type=int, default=1, help='alpha')

	parser.add_argument('--beta', type=int, default=1, help='beta')

	parser.add_argument('-Layers', action='store', dest='Layers', default={
		'Layer_1': [64, 40],
		'Layer_2': [64, 40],
		'Layer_3': [64, 40],
		'Layer_4': [64, 40],
	})

	args = parser.parse_args()

	return args