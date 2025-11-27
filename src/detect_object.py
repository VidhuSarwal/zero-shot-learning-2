#
# detect_object.py
#
# Created by Samet Cetin.
# Contact: cetin.samet@outlook.com
#

import os
import sys
import numpy as np
from PIL import Image

from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize

from feature_extractor import get_model, get_features
from train import load_keras_model

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

WORD2VECPATH    = "../data/class_vectors.npy"
MODELPATH       = "../model/"

def main(argv):

    if len(argv) != 1:
        print("Usage: python3 detect_object.py input-image-path")
        exit()

    # READ IMAGE
    IMAGEPATH = argv[0]
    try:
        img = Image.open(IMAGEPATH).resize((224, 224))
    except Exception as e:
        print(f"Error loading image: {e}")
        exit(1)

    print(f"\n🔍 Analyzing image: {IMAGEPATH}")
    print("=" * 50)

    # LOAD PRETRAINED VGG16 MODEL FOR FEATURE EXTRACTION
    print("Loading VGG16 feature extractor...")
    vgg_model   = get_model()
    
    # EXTRACT IMAGE FEATURE
    print("Extracting image features...")
    img_feature = get_features(vgg_model, img)
    
    # L2 NORMALIZE FEATURE
    img_feature = normalize(img_feature, norm='l2')

    # LOAD ZERO-SHOT MODEL
    print("Loading zero-shot model...")
    model       = load_keras_model(model_path=MODELPATH)
    
    # MAKE PREDICTION
    print("Making predictions...")
    pred        = model.predict(img_feature, verbose=0)

    # LOAD CLASS WORD2VECS (fixed: add allow_pickle=True and fix dtype)
    class_vectors       = sorted(np.load(WORD2VECPATH, allow_pickle=True), key=lambda x: x[0])
    classnames, vectors = zip(*class_vectors)
    classnames          = list(classnames)
    vectors             = np.asarray(vectors, dtype=np.float32)

    # PLACE WORD2VECS IN KDTREE
    tree                = KDTree(vectors)
    
    # FIND CLOSEST WORD2VEC and GET PREDICTION RESULT
    dist, index         = tree.query(pred, k=5)
    pred_labels         = [classnames[idx] for idx in index[0]]
    distances           = dist[0]

    # PRINT RESULT
    print("\n" + "=" * 50)
    print("📊 TOP-5 PREDICTIONS")
    print("=" * 50)
    for i, (classname, distance) in enumerate(zip(pred_labels, distances)):
        confidence = max(0, 100 * (1 - distance / distances[-1]))
        print(f"{i+1}. {classname.upper():<15} (confidence: {confidence:.1f}%)")
    print("=" * 50)
    
    # Print class categories
    print("\n📝 Note: Predictions are from these categories:")
    print("   Training classes: arm, boy, bread, chicken, child, computer,")
    print("                     ear, house, leg, sandwich, television,")
    print("                     truck, vehicle, watch, woman")
    print("   Zero-shot classes: car, food, hand, man, neck")
    print()
    
    return

if __name__ == '__main__':
    main(sys.argv[1:])
