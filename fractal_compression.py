import sys
import time
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim

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

def compress(img, source_size, destination_size, step, list):
	transforms = []
	transformed_blocks = generate_all_transformed_blocks(img, source_size, destination_size, step)
	for i in range(img.shape[0] // destination_size):
		transforms.append([])
		for j in range(img.shape[1] // destination_size):
			transforms[i].append(None)
			min_d = float('inf')
			D = img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size]
			for m in list:
				k, l, direction, angle, S = transformed_blocks[m]
				contrast, brightness = find_contrast_and_brightness2(D, S)
				S = contrast*S + brightness
				d = np.sum(np.square(D - S))
				if d < min_d:
					min_d = d
					transforms[i][j] = (k, l, direction, angle, contrast, brightness)
	return transforms

def decompress(transforms, source_size, destination_size, step, nb_iter=6):
	factor = source_size // destination_size
	height = len(transforms) * destination_size
	width = len(transforms[0]) * destination_size
	iterations = [np.random.randint(0, 256, (height, width))]
	cur_img = np.zeros((height, width))
	for i_iter in range(nb_iter):
		for i in range(len(transforms)):
			for j in range(len(transforms[i])):
				k, l, flip, angle, contrast, brightness = transforms[i][j]
				S = reduce(iterations[-1][k*step:k*step+source_size,l*step:l*step+source_size], factor)
				D = apply_transform(S, flip, angle, contrast, brightness)
				cur_img[i*destination_size:(i+1)*destination_size,j*destination_size:(j+1)*destination_size] = D
		iterations.append(cur_img)
		cur_img = np.zeros((height, width))
	return iterations

directions = [1, -1]
angles = [0, 90, 180, 270]
candidates = list(zip(directions, angles))

def args_parser():
    args = sys.argv
    if len(args) < 5:
        print "Not enough files"
        sys.exit(1)
    return args[1], args[2], args[3], args[4]

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def psnr(imageA, imageB):
  mse = np.mean( (imageA - imageB) ** 2 )
  if mse == 0:
      return 100
  PIXEL_MAX = 255.0
  return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def complete_test(method_name, list, img):
	print("%s:" % method_name)
	start_time = time.time()
	transforms = compress(img, 8, 4, 8, list)
	iterations = decompress(transforms, 8, 4, 8)
	img2 = iterations[len(iterations) - 1]
	plt.figure()
	plt.imshow(img2, cmap='gray', interpolation='none')
	plt.title(method_name)
	m = mse(img, img2)
	s = ssim(np.array(img), np.array(img2))
	p = psnr(img, img2)
	print("time = %.5f" % (time.time() - start_time))
	print("MSE = %s, SSIM = %.3f, PSNR = %s" % (m, s, p))

if __name__ == '__main__':
	imgPath, congruent_file, zikkurat_file, mersenne_file = args_parser()
	img = mpimg.imread(imgPath)
	img = get_greyscale_image(img)
	img = reduce(img, 4)
	plt.figure()
	plt.imshow(img, cmap='gray', interpolation='none')
	cong = np.fromfile(congruent_file, int, -1, ' ').tolist()
	zikk = np.fromfile(zikkurat_file, int, -1, ' ').tolist()
	mers = np.fromfile(mersenne_file, int, -1, ' ').tolist()
	complete_test('Конгруэнтный', cong, img)
	complete_test('Зиккурат', zikk, img)
	complete_test('Мерсенн', mers, img)
	plt.show()
