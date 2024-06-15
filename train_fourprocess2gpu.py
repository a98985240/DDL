import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from mpi4py import MPI
import numpy as np
import logging
import random
import argparse

# 設置命令行參數
parser = argparse.ArgumentParser(description='Train a model with MPI and TensorFlow.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model.')
args = parser.parse_args()

# Set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Suppress TensorFlow warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Configure logging
log_file = 'training.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file)
])
logger = logging.getLogger()

# 添加只記錄到檔案的處理程序
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 添加只打印到終端機的處理程序
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
console_handler.addFilter(lambda record: "gradient" not in record.getMessage())  # 過濾包含 "gradient" 的日誌
logger.addHandler(console_handler)

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def create_model():
    logger.info("Creating model...")
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    logger.info("Model created.")
    return model

def create_dataset(x_train, y_train, batch_size):
    logger.info("Creating dataset...")
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(buffer_size=1024, seed=seed).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    logger.info("Dataset created.")
    return dataset

def allreduce_gradients(gradients, comm, size):
    averaged_gradients = []
    for grad in gradients:
        tensor = grad.numpy()
        comm.Allreduce(MPI.IN_PLACE, tensor, op=MPI.SUM)
        tensor /= size
        averaged_gradients.append(tf.convert_to_tensor(tensor))
    return averaged_gradients

def train_model(rank, size, epochs):
    # Set CUDA device to the appropriate GPU based on rank
    if rank in [0, 1]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif rank in [2, 3]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # 打印當前進程和所使用的 GPU
    logger.info(f"Process {rank} is using GPU {os.environ['CUDA_VISIBLE_DEVICES']}")

    logger.info(f"Rank {rank}/{size} is starting training...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize data
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

    batch_size = 64 // size
    train_dataset = create_dataset(x_train, y_train, batch_size)

    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])

    comm = MPI.COMM_WORLD
    steps_per_epoch = len(x_train) // (batch_size * size)

    for epoch in range(epochs):
        if rank == 0:
            print(f"Epoch {epoch+1} started...")
        logger.info(f"Epoch {epoch+1} started...")
        for step, (x_batch, y_batch) in enumerate(train_dataset.take(steps_per_epoch)):
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = loss_object(y_batch, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # 記錄平均前的梯度
            if step == 0:  # Log gradients only for the first step of each epoch for simplicity
                for i, grad in enumerate(gradients):
                    logger.info(f"Rank {rank} gradient {i} before allreduce: {np.mean(grad.numpy())}")

            averaged_gradients = allreduce_gradients(gradients, comm, size)
            
            # 記錄平均後的梯度
            if step == 0:  # Log averaged gradients for the first step of each epoch
                for i, avg_grad in enumerate(averaged_gradients):
                    logger.info(f"Rank {rank} averaged gradient {i} after allreduce: {np.mean(avg_grad.numpy())}")

            optimizer.apply_gradients(zip(averaged_gradients, model.trainable_variables))
        
        logger.info(f"Epoch {epoch+1} ended.")

        if rank == 0:
            loss, acc = model.evaluate(x_test, y_test, verbose=0)
            print(f"Epoch {epoch+1}, Loss: {loss}, Accuracy: {acc}")
            logger.info(f"Epoch {epoch+1}, Loss: {loss}, Accuracy: {acc}")

            # Print predictions for a few test samples
            sample_predictions = model.predict(x_test[:5])
            predicted_labels = np.argmax(sample_predictions, axis=1)
            true_labels = y_test[:5].flatten()
            logger.info(f"Sample Predictions: {predicted_labels}")
            logger.info(f"True Labels: {true_labels}")

    logger.info(f"Rank {rank}/{size} finished training.")

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Set CUDA device to the appropriate GPU based on rank
    if rank in [0, 1]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif rank in [2, 3]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if size == 1:
        print(f"Running with a single MPI process (Rank {rank}/{size})")
    else:
        print(f"Running with multiple MPI processes (Rank {rank}/{size})")

    if size < 1:
        logger.error("This program requires at least 1 MPI process.")
        exit()

    # Print out the GPU each process is using
    logger.info(f"Process {rank} is using GPU {os.environ['CUDA_VISIBLE_DEVICES']}")

    train_model(rank, size, args.epochs)
