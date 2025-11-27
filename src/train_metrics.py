import numpy as np
np.random.seed(123)
import gzip
import _pickle as cPickle
import os
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import KDTree

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical


WORD2VECPATH    = "../data/class_vectors.npy"
DATAPATH        = "../data/zeroshot_data.pkl"
MODELPATH       = "../model/"

def load_keras_model(model_path):
    with open(model_path +"model.json", 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_path+"model.h5")
    return loaded_model

def save_keras_model(model, model_path):
    """save Keras model and its weights"""
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_json = model.to_json()
    with open(model_path + "model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(model_path + "model.h5")
    print("-> zsl model is saved.")
    return

def load_data():
    """read data, create datasets"""
    with gzip.GzipFile(DATAPATH, 'rb') as infile:
        data = cPickle.load(infile)

    label_encoder   = LabelEncoder()
    label_encoder.fit(train_classes)

    training_data = [instance for instance in data if instance[0] in train_classes]
    zero_shot_data = [instance for instance in data if instance[0] not in train_classes]
    np.random.shuffle(training_data)

    train_size  = 300
    train_data  = list()
    valid_data  = list()
    for class_label in train_classes:
        ct = 0
        for instance in training_data:
            if instance[0] == class_label:
                if ct < train_size:
                    train_data.append(instance)
                    ct+=1
                    continue
                valid_data.append(instance)

    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)

    train_data = [(instance[1], to_categorical(label_encoder.transform([instance[0]]), num_classes=15))for instance in train_data]
    valid_data = [(instance[1], to_categorical(label_encoder.transform([instance[0]]), num_classes=15)) for instance in valid_data]

    x_train, y_train    = zip(*train_data)
    x_train, y_train    = np.squeeze(np.asarray(x_train)), np.squeeze(np.asarray(y_train))
    x_train = normalize(x_train, norm='l2')

    x_valid, y_valid = zip(*valid_data)
    x_valid, y_valid = np.squeeze(np.asarray(x_valid)), np.squeeze(np.asarray(y_valid))
    x_valid = normalize(x_valid, norm='l2')

    y_zsl, x_zsl = zip(*zero_shot_data)
    x_zsl, y_zsl = np.squeeze(np.asarray(x_zsl)), np.squeeze(np.asarray(y_zsl))
    x_zsl = normalize(x_zsl, norm='l2')

    print("-> data loading is completed.")
    return (x_train, x_valid, x_zsl), (y_train, y_valid, y_zsl)


def custom_kernel_init(shape):
    class_vectors       = np.load(WORD2VECPATH)
    training_vectors    = sorted([(label, vec) for (label, vec) in class_vectors if label in train_classes], key=lambda x: x[0])
    classnames, vectors = zip(*training_vectors)
    vectors             = np.asarray(vectors, dtype=np.float)
    vectors             = vectors.T
    return vectors

def build_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(4096,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(NUM_ATTR, activation='relu'))
    model.add(Dense(NUM_CLASS, activation='softmax', trainable=False, kernel_initializer=custom_kernel_init))

    print("-> model building is completed.")
    return model


def train_model(model, train_data, valid_data):
    x_train, y_train = train_data
    x_valid, y_valid = valid_data
    adam = Adam(lr=5e-5)
    model.compile(loss      = 'categorical_crossentropy',
                  optimizer = adam,
                  metrics   = ['categorical_accuracy', 'top_k_categorical_accuracy'])

    history = model.fit(x_train, y_train,
                        validation_data = (x_valid, y_valid),
                        verbose         = 2,
                        epochs          = EPOCH,
                        batch_size      = BATCH_SIZE,
                        shuffle         = True)

    print("-> model training is completed.")
    return history

def plot_training_history(history):
    """Plot and save training metrics"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Zero-Shot Learning Training Metrics', fontsize=16)
    
    # Plot 1: Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Categorical Accuracy
    axes[0, 1].plot(history.history['categorical_accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Categorical Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Top-K Accuracy
    axes[1, 0].plot(history.history['top_k_categorical_accuracy'], label='Training Top-K Accuracy')
    axes[1, 0].plot(history.history['val_top_k_categorical_accuracy'], label='Validation Top-K Accuracy')
    axes[1, 0].set_title('Top-K Categorical Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Summary Statistics
    axes[1, 1].axis('off')
    final_metrics = f"""
    Final Training Metrics:
    
    Training Loss: {history.history['loss'][-1]:.4f}
    Validation Loss: {history.history['val_loss'][-1]:.4f}
    
    Training Accuracy: {history.history['categorical_accuracy'][-1]:.4f}
    Validation Accuracy: {history.history['val_categorical_accuracy'][-1]:.4f}
    
    Training Top-K Acc: {history.history['top_k_categorical_accuracy'][-1]:.4f}
    Validation Top-K Acc: {history.history['val_top_k_categorical_accuracy'][-1]:.4f}
    
    Best Validation Accuracy: {max(history.history['val_categorical_accuracy']):.4f}
    at Epoch: {np.argmax(history.history['val_categorical_accuracy']) + 1}
    """
    axes[1, 1].text(0.1, 0.5, final_metrics, fontsize=12, verticalalignment='center', 
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('../training_metrics.png', dpi=300, bbox_inches='tight')
    print("-> Training metrics plot saved as 'training_metrics.png'")
    plt.show()
    
    # Save history to file
    np.save('../training_history.npy', history.history)
    print("-> Training history saved as 'training_history.npy'")

def main():

    global train_classes
    with open('train_classes.txt', 'r') as infile:
        train_classes = [str.strip(line) for line in infile]

    global zsl_classes
    with open('zsl_classes.txt', 'r') as infile:
        zsl_classes = [str.strip(line) for line in infile]

    global NUM_CLASS, NUM_ATTR, EPOCH, BATCH_SIZE
    NUM_CLASS = 15
    NUM_ATTR = 300
    BATCH_SIZE = 128
    EPOCH = 65

    # TRAINING PHASE
    (x_train, x_valid, x_zsl), (y_train, y_valid, y_zsl) = load_data()
    model = build_model()
    history = train_model(model, (x_train, y_train), (x_valid, y_valid))
    print(model.summary())
    
    # PLOT TRAINING METRICS
    plot_training_history(history)

    # CREATE AND SAVE ZSL MODEL
    inp         = model.input
    out         = model.layers[-2].output
    zsl_model   = Model(inp, out)
    print(zsl_model.summary())
    save_keras_model(zsl_model, model_path=MODELPATH)

    # EVALUATION OF ZERO-SHOT LEARNING PERFORMANCE
    class_vectors       = sorted(np.load(WORD2VECPATH), key=lambda x: x[0])
    classnames, vectors = zip(*class_vectors)
    classnames          = list(classnames)
    vectors             = np.asarray(vectors, dtype=np.float)

    tree        = KDTree(vectors)
    pred_zsl    = zsl_model.predict(x_zsl)

    top5, top3, top1 = 0, 0, 0
    for i, pred in enumerate(pred_zsl):
        pred            = np.expand_dims(pred, axis=0)
        dist_5, index_5 = tree.query(pred, k=5)
        pred_labels     = [classnames[index] for index in index_5[0]]
        true_label      = y_zsl[i]
        if true_label in pred_labels:
            top5 += 1
        if true_label in pred_labels[:3]:
            top3 += 1
        if true_label in pred_labels[0]:
            top1 += 1

    print()
    print("=" * 50)
    print("ZERO SHOT LEARNING EVALUATION")
    print("=" * 50)
    print("-> Top-5 Accuracy: %.2f%%" % (100 * top5 / float(len(x_zsl))))
    print("-> Top-3 Accuracy: %.2f%%" % (100 * top3 / float(len(x_zsl))))
    print("-> Top-1 Accuracy: %.2f%%" % (100 * top1 / float(len(x_zsl))))
    print("=" * 50)
    return

if __name__ == '__main__':
    main()
