#!/usr/bin/env python

import os
from glob import glob

from keras import layers, models
import tensorflow as tf
import numpy as np
from PIL import Image

from constants import TILES_DIR, NN_MODEL_PATH, FEN_CHARS, USE_GRAYSCALE

RATIO = 0.82
N_EPOCHS = 20

CACHE_TRAIN_X = 'train_images.npy'
CACHE_TRAIN_Y = 'train_labels.npy'
CACHE_TEST_X  = 'test_images.npy'
CACHE_TEST_Y  = 'test_labels.npy'


def image_data(image_path):
	img = Image.open(image_path)

	if USE_GRAYSCALE:
		img = img.convert('L')
	else:
		img = img.convert('RGB')

	img = img.resize((32, 32))

	arr = np.array(img).astype(np.float32) / 255.0

	if USE_GRAYSCALE:
		arr = np.expand_dims(arr, axis=-1)

	return arr


def create_model():
	input_shape = (32, 32, 1) if USE_GRAYSCALE else (32, 32, 3)

	model = models.Sequential([
		layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
		layers.MaxPooling2D((2, 2)),

		layers.Conv2D(64, (3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),

		layers.Conv2D(64, (3, 3), activation='relu'),

		layers.Flatten(),
		layers.Dense(64, activation='relu'),
		layers.Dense(len(FEN_CHARS), activation='softmax'),
	])

	model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

	return model


def load_or_build_dataset():
	# ✅ FAST PATH: load cache if exists
	if all(os.path.exists(p) for p in [
		CACHE_TRAIN_X, CACHE_TRAIN_Y,
		CACHE_TEST_X, CACHE_TEST_Y
	]):
		print("Loading dataset from cache...")

		train_images = np.load(CACHE_TRAIN_X)
		train_labels = np.load(CACHE_TRAIN_Y)
		test_images  = np.load(CACHE_TEST_X)
		test_labels  = np.load(CACHE_TEST_Y)

		return (train_images, train_labels), (test_images, test_labels)

	# ❗ SLOW PATH (first run only)
	print("Building dataset (first run, caching enabled)...")

	all_paths = np.array(glob('{}/*/*/*.png'.format(TILES_DIR)))

	np.random.seed(1)
	np.random.shuffle(all_paths)

	divider = int(len(all_paths) * RATIO)
	train_paths = all_paths[:divider]
	test_paths  = all_paths[divider:]

	train_images, train_labels = [], []
	for image_path in train_paths:
		piece_type = image_path[-5]
		assert piece_type in FEN_CHARS

		train_images.append(image_data(image_path))
		train_labels.append(FEN_CHARS.index(piece_type))

	test_images, test_labels = [], []
	for image_path in test_paths:
		piece_type = image_path[-5]
		assert piece_type in FEN_CHARS

		test_images.append(image_data(image_path))
		test_labels.append(FEN_CHARS.index(piece_type))

	train_images = np.array(train_images)
	train_labels = np.array(train_labels)
	test_images  = np.array(test_images)
	test_labels  = np.array(test_labels)

	print("Saving dataset cache...")

	np.save(CACHE_TRAIN_X, train_images)
	np.save(CACHE_TRAIN_Y, train_labels)
	np.save(CACHE_TEST_X,  test_images)
	np.save(CACHE_TEST_Y,  test_labels)

	return (train_images, train_labels), (test_images, test_labels)


if __name__ == '__main__':
	print('TensorFlow {}'.format(tf.__version__))

	(train_images, train_labels), (test_images, test_labels) = load_or_build_dataset()

	if not len(train_images):
		print("No training images found!")
		exit(1)

	model = create_model()

	model.fit(
		train_images,
		train_labels,
		epochs=N_EPOCHS,
		validation_data=(test_images, test_labels)
	)
	print('Saving CNN model to {}'.format(NN_MODEL_PATH))
	svPath, xt = os.path.splitext( NN_MODEL_PATH.rstrip('/'))

	svPath += '.h5' if xt.lower() in ['.tf', '.pb', '.pbtxt', '.h5', '.hdf5'] else ext +'.h5'

	print('Evaluating CNN model on test data:')
	model.evaluate( test_images, test_labels, verbose=1)


