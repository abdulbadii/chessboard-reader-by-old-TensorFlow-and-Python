#!/usr/bin/env python

import sys
import os
from glob import glob
from io import BytesIO
from functools import reduce

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import models
import numpy as np
from PIL import Image

from constants import (
	TILES_DIR, NN_MODEL_PATH, FEN_CHARS, USE_GRAYSCALE, DETECT_CORNERS )

from utils import compressed_fen
from chessboard_finder import get_chessboard_corners
from chessboard_image import get_chessboard_tiles

OUT_FILE = "debug.html"

# Image preprocessing (FAST + SAFE)
# -----------------------------
def preprocess_pil_image( PImg):
	if USE_GRAYSCALE:
		PImg = PImg.convert('L')
	else:
		PImg = PImg.convert('RGB')

	PImg = PImg.resize((32, 32))

	arr = np.array(PImg).astype(np.float32) / 255.0

	if USE_GRAYSCALE:
		arr = np.expand_dims(arr, axis=-1)

	return arr


# Extract tiles from chessboard
# -----------------------------
def _chessboard_tiles_img_data(chessboard_img_path, options=None):
	n_channels = 1 if USE_GRAYSCALE else 3

	tiles = get_chessboard_tiles(
		chessboard_img_path,
		use_grayscale=USE_GRAYSCALE
	)

	img_data_list = []

	for i in range(64):
		buf = BytesIO()
		tiles[i].save(buf, format='PNG')
		buf.seek(0)

		PImg = Image.open(buf)

		arr = preprocess_pil_image(PImg)

		img_data_list.append(arr)

	return img_data_list


# Color based on confidence
# -----------------------------
def _confidence_color(confidence):
	if confidence >= 0.999:
		return "#00C176"
	elif confidence > 0.99:
		return "#88C100"
	elif confidence > 0.95:
		return "#FABE28"
	elif confidence > 0.9:
		return "#FF8A00"
	else:
		return "#FF003C"


# Save debug HTML
# -----------------------------
def _save_output_html(chessboard_img_path, fen, predictions, confidence):
	conf_color = _confidence_color(confidence)

	html = '<h3>{}</h3>'.format(chessboard_img_path)
	html += '<div class="boards-row">'

	html += '<img src="{}" />'.format(chessboard_img_path)
	html += '<img src="http://www.fen-to-image.com/image/32/{}"/>'.format(fen)

	html += '<div class="predictions-matrix">'

	for i in range(8):
		html += '<div>'
		for j in range(8):
			c = predictions[i * 8 + j]
			html += '<div class="prediction" style="color: {}">{}</div>'.format(
				_confidence_color(c),
				format(c, '.3f')
			)
		html += '</div>'

	html += '</div></div><br />'

	html += '<a href="https://lichess.org/editor/{}" target="_blank">{}</a>'.format(
		fen, fen
	)

	html += '<div style="color: {}">{}</div>'.format(conf_color, confidence)
	html += '<br /><br />'

	with open(OUT_FILE, "a") as f:
		f.write(html)


# Predict one tile
# -----------------------------
def predict_tile(tile_img_data):
	probs = model.predict(np.array([tile_img_data]), verbose=0)[0]

	idx = int(np.argmax(probs))
	conf = float(probs[idx])

	return FEN_CHARS[idx], conf


# Predict full chessboard
# -----------------------------
def predict_chessboard(chessboard_img_path, options):
	if not options.quiet:
		print("Predicting:", chessboard_img_path)

	img_data_list = _chessboard_tiles_img_data(chessboard_img_path, options)

	predictions = []

	for tile in img_data_list:
		fen_char, prob = predict_tile(tile)

		if not options.quiet:	print((fen_char, prob))

		predictions.append((fen_char, prob))

	fen = compressed_fen(
		'/'.join(
			[''.join(r) for r in np.reshape(
				[p[0] for p in predictions],
				[8, 8]
			)]
		)
	)
	confidence = reduce(lambda x, y: x * y, [p[1] for p in predictions])

	if not options.quiet:
		print("Confidence:", confidence)

	print("https://lichess.org/editor/{}".format(fen))

	_save_output_html(
		chessboard_img_path,
		fen,
		[p[1] for p in predictions],
		confidence
	)

	print("Saved to", OUT_FILE)

	return fen

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-q", "--quiet", action="store_true")
	parser.add_argument("-d", "--debug", action="store_true")
	parser.add_argument("image_path")

	args = parser.parse_args()

	if len(sys.argv) == 1: exit()
	print('TensorFlow', __import__('tensorflow').__version__)

	nnPath, xt = os.path.splitext(NN_MODEL_PATH.rstrip('/'))

	nnPath += '.h5' if xt.lower() in ['.tf', '.pb', '.pbtxt', '.h5', '.hdf5'] else xt +'.h5'

	model = models.load_model( nnPath)

	with open( OUT_FILE, "w") as f:
		f.write('<link rel="stylesheet" href="./web/style.css" />')

	for chessboard_image_path in sorted( glob(args.image_path)):
		print( predict_chessboard(chessboard_image_path, args))