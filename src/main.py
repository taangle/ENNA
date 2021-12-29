from datetime import datetime
import keras

from src.evolver import Evolver
from src.histories import initialize_history_dir

import tensorflow as tf

POP_SIZE = 8
MAX_GEN = 4
GEN_TO_START_BEING_ELITE = 30
PARENT_COUNT = 4

BATCH_SIZE = 64
EPOCHS = 8
MIN_FILTERS = 2
MAX_FILTERS = 16
MIN_KERNEL_DIM = 1
MAX_KERNEL_DIM = 4
MIN_STRIDE_DIM = 1
MAX_STRIDE_DIM = 4
MIN_HIDDEN_LAYERS = 2
MAX_HIDDEN_LAYERS = 16
MIN_UNITS = 8
MAX_UNITS = 1024
MAX_DROPOUT = .5


def main():
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")

    print(f"GPUs:\n{tf.config.list_physical_devices('GPU')}")

    (x_train, y_train), (x_test, y_test), num_classes, input_shape = load_data()
    data = (x_train, y_train), (x_test, y_test)

    evolver = Evolver(
        data=data,
        start_time_str=start_time_str,
        pop_size=POP_SIZE,
        max_gen=MAX_GEN,
        gen_to_start_being_elite=GEN_TO_START_BEING_ELITE,
        parent_count=PARENT_COUNT,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        input_shape=input_shape,
        num_classes=num_classes,
        min_filters=MIN_FILTERS,
        max_filters=MAX_FILTERS,
        min_kernel_dim=MIN_KERNEL_DIM,
        max_kernel_dim=MAX_KERNEL_DIM,
        min_stride_dim=MIN_STRIDE_DIM,
        max_stride_dim=MAX_STRIDE_DIM,
        min_hidden_layers=MIN_HIDDEN_LAYERS,
        max_hidden_layers=MAX_HIDDEN_LAYERS,
        min_units=MIN_UNITS,
        max_units=MAX_UNITS,
        max_dropout=MAX_DROPOUT
    )

    initialize_history_dir(evolver)

    # generate population
    population = evolver.generate_random_population()

    # evolve the population
    population = evolver.evolve(population)

    # now have an evolved population
    print("\nThe champions:")
    max_fitness = 0
    max_fitness_genome = None
    for genome in population:
        print(genome)
        if genome.fitness > max_fitness:
            max_fitness = genome.fitness
            max_fitness_genome = genome

    print("\nThe ultimate champion:")
    print(max_fitness_genome)


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # uncomment the below to make things run faster for testing
    # x_train = x_train[0:len(x_train) // 4]
    # y_train = y_train[0:len(y_train) // 4]
    # x_test = x_test[0:len(x_test) // 4]
    # y_test = y_test[0:len(y_test) // 4]
    num_classes = 10
    input_shape = (32, 32, 3)

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test), num_classes, input_shape


if __name__ == "__main__":
    main()
