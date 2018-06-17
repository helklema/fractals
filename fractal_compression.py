import sys
import time
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np

def get_greyscale_image(img):
	return np.mean(img[:,:,:2], 2)

def reduce(img, factor):
	result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
	for i in range(result.shape[0]):
		for j in range(result.shape[1]):
			result[i,j] += np.mean(img[i*factor:(i+1)*factor,j*factor:(j+1)*factor])
	return result

def rotate(img, angle):
	return ndimage.rotate(img, angle, reshape=False)

def flip(img, direction):
	return img[::direction,:]

def apply_transform(img, direction, angle, contrast=1.0, brightness=0.0):
	return contrast*rotate(flip(img, direction), angle) + brightness

def find_contrast_and_brightness2(D, S):
	A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
	b = np.reshape(D, (D.size,))
	x, _, _, _ = np.linalg.lstsq(A, b, -1)
	return x[1], x[0]

def generate_all_transformed_blocks(img, source_size, destination_size, step):
	factor = source_size // destination_size
	transformed_blocks = []
	for k in range((img.shape[0] - source_size) // step + 1):
		for l in range((img.shape[1] - source_size) // step + 1):
			S = reduce(img[k*step:k*step+source_size,l*step:l*step+source_size], factor)
			for direction, angle in candidates:
				transformed_blocks.append((k, l, direction, angle, apply_transform(S, direction, angle)))
	return transformed_blocks

def compress(img, source_size, destination_size, step):
	transforms = []
	transformed_blocks = generate_all_transformed_blocks(img, source_size, destination_size, step)
	for i in range(img.shape[0] // destination_size):
		transforms.append([])
		for j in range(img.shape[1] // destination_size):
			transforms[i].append(None)
			min_d = float('inf')
			D = img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size]
			for k, l, direction, angle, S in transformed_blocks:
				contrast, brightness = find_contrast_and_brightness2(D, S)
				S = contrast*S + brightness
				d = np.sum(np.square(D - S))
				if d < min_d:
					min_d = d
					transforms[i][j] = (k, l, direction, angle, contrast, brightness)
	return transforms

def decompress(transforms, source_size, destination_size, step, iterations, nb_iter=3):
	factor = source_size // destination_size
	height = len(transforms) * destination_size
	width = len(transforms[0]) * destination_size
	# iterations = [np.random.randint(0, 256, (height, width))]
	# print(iterations) # type-list
	cur_img = np.zeros((height, width))
	for i_iter in range(nb_iter):
		start_time = time.time()
		for i in range(len(transforms)):
			for j in range(len(transforms[i])):
				k, l, flip, angle, contrast, brightness = transforms[i][j]
				S = reduce(iterations[-1][k*step:k*step+source_size,l*step:l*step+source_size], factor)
				D = apply_transform(S, flip, angle, contrast, brightness)
				cur_img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size] = D
		iterations.append(cur_img)
		cur_img = np.zeros((height, width))
		print("time = %.5f" % (time.time() - start_time))
	return iterations

def plot_iterations(iterations, target=None):
	plt.figure()
	nb_row = math.ceil(np.sqrt(len(iterations)))
	nb_cols = nb_row
	for i, img in enumerate(iterations):
		plt.subplot(nb_row, nb_cols, i+1)
		plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
		if target is None:
			plt.title(str(i))
		else:
			plt.title(str(i) + ' (' + '{0:.2f}'.format(np.sqrt(np.mean(np.square(target - img)))) + ')')
		frame = plt.gca()
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)
	plt.tight_layout()


directions = [1, -1]
angles = [0, 90, 180, 270]
candidates = list(zip(directions, angles))

def args_parser():
    args = sys.argv
    if len(args) < 5:
        print "Not enough files"
        sys.exit(1)
    return args[1], args[2], args[3], args[4]
					
if __name__ == '__main__':
	imgPath, congruentFile, zikkuratFile, mersenneFile = args_parser()
	cong = np.fromfile(congruentFile, int, -1, ' ')
	iterations = np.array(map(lambda item: item.tolist(), np.split(cong, 10)))
	# zikk = open(zikkuratFile, 'r').read().split()
	# mers = open(mersenneFile, 'r').read().split()
	img = mpimg.imread(imgPath)
	img = get_greyscale_image(img)
	img = reduce(img, 4)
	plt.figure()
	plt.imshow(img, cmap='gray', interpolation='none')
	transforms = compress(img, 8, 4, 8)
	iterations = decompress(transforms, 8, 4, 8, iterations)
	plot_iterations(iterations, img)
	plt.show()