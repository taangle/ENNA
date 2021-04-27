import csv
import random
from pathlib import Path
from datetime import datetime
import keras
from genome import generate_random_population, combine_genomes

POP_SIZE = 64
MAX_GEN = 256
GEN_TO_START_BEING_ELITE = 200
PARENT_COUNT = 24

# TODO learned batches?
BATCH_SIZE = 128
EPOCHS = 15
MIN_FILTERS = 2
MAX_FILTERS = 32
MIN_KERNEL_DIM = 1
MAX_KERNEL_DIM = 4
MIN_STRIDE_DIM = 1
MAX_STRIDE_DIM = 4
MIN_HIDDEN_LAYERS = 2
MAX_HIDDEN_LAYERS = 8
MIN_UNITS = 10
MAX_UNITS = 512
MAX_DROPOUT = .5

# uncomment the following (and comment out the above) in order to run the program quickly for testing purposes
# there are also some lines in load_data you can uncomment to make it even faster
# POP_SIZE = 4
# MAX_GEN = 5
# GEN_TO_START_BEING_ELITE = 4
# PARENT_COUNT = 2
# BATCH_SIZE = 50
# EPOCHS = 4
# MIN_FILTERS = 2
# MAX_FILTERS = 8
# MIN_KERNEL_DIM = 1
# MAX_KERNEL_DIM = 4
# MIN_STRIDE_DIM = 1
# MAX_STRIDE_DIM = 4
# MIN_HIDDEN_LAYERS = 2
# MAX_HIDDEN_LAYERS = 4
# MIN_UNITS = 10
# MAX_UNITS = 32
# MAX_DROPOUT = .5


def main():
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    initialize_history_dir(start_time_str)

    (x_train, y_train), (x_test, y_test), num_classes, input_shape = load_data()
    data = (x_train, y_train), (x_test, y_test)

    # generate population
    pop = generate_random_population(POP_SIZE,
                                     input_shape=input_shape,
                                     batch_size=BATCH_SIZE,
                                     epochs=EPOCHS,
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
                                     max_dropout=MAX_DROPOUT,
                                     num_classes=num_classes)

    # test fitness (opportunity for parallelization)
    for genome in pop:
        genome.calculate_fitness(data)
        print(genome)
    record_generation(0, pop, start_time_str)
    print("==End of generation 0==")

    for gen_number in range(MAX_GEN):
        # select parents with some strategy
        if gen_number >= GEN_TO_START_BEING_ELITE:
            parents = select_parents(pop, PARENT_COUNT, be_elite=True)
        else:
            parents = select_parents(pop, PARENT_COUNT)

        # refill population with mutated children of parents
        pop = []
        while len(pop) < POP_SIZE:
            random.shuffle(parents)
            # if PARENT_COUNT is odd, one parent will get left out, but not necessarily indefinitely
            pairs = zip(parents[::2], parents[1::2])
            for pair in pairs:
                pop.append(combine_genomes((pair[0]), pair[1],
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
                                           max_dropout=MAX_DROPOUT,
                                           num_classes=num_classes
                                           ))
                if len(pop) == POP_SIZE:
                    break

        # test fitness
        for genome in pop:
            if genome.fitness is None:
                genome.calculate_fitness(data)

            print(genome)

        record_generation(gen_number + 1, pop, start_time_str)
        print(f"==End of generation {gen_number + 1}==")

    # now you have an evolved population
    print("\nThe champions:")
    max_fitness = 0
    max_fitness_genome = None
    for genome in pop:
        print(genome)
        if genome.fitness > max_fitness:
            max_fitness = genome.fitness
            max_fitness_genome = genome

    print("\nThe ultimate champion:")
    print(max_fitness_genome)


# TODO just use a DB instead
def initialize_history_dir(start_time_str):
    Path(f"histories/{start_time_str}").mkdir(parents=True, exist_ok=True)

    with open(f"histories/{start_time_str}/info.csv", mode="w", newline="") as info_file:
        info_writer = csv.writer(info_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        # TODO add the rest of the params
        info_writer.writerow(["batch size", "epochs", "pop_size", "max_gen", "gen_to_start_being_elite", "parent_count", "min units", "max units"])
        info_writer.writerow([BATCH_SIZE, EPOCHS, POP_SIZE, MAX_GEN, GEN_TO_START_BEING_ELITE, PARENT_COUNT, MIN_UNITS, MAX_UNITS])

    with open(f"histories/{start_time_str}/history.csv", mode="a", newline="") as history_file:
        history_writer = csv.writer(history_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        history_writer.writerow(["generation", "genome", "accuracy", "training_time", "layer", "type", "units", "activation"])

def record_generation(gen_number, pop, start_time_str):
    with open(f"histories/{start_time_str}/history.csv", mode="a", newline="") as history_file:
        history_writer = csv.writer(history_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for genome_index in range(len(pop)):
            genome = pop[genome_index]
            for layer_index in range(len(genome.layers)):
                layer = genome.layers[layer_index]
                # TODO add the other layers
                if layer["type"] == "Dense":
                    history_writer.writerow([
                        gen_number,
                        genome_index,
                        genome.accuracy,
                        genome.training_time,
                        layer_index,
                        layer["type"],
                        layer["units"],
                        layer["activation"]
                    ])


def select_parents(pop, parent_count, be_elite=False):
    total_fitness = sum(genome.fitness for genome in pop)
    probability_dict = {}
    previous_prob = 0
    max_fitness = 0
    max_fitness_genome = None
    for i in range(len(pop)):
        genome = pop[i]
        probability_dict[i] = previous_prob + (genome.fitness / total_fitness)
        previous_prob = probability_dict[i]
        if genome.fitness > max_fitness:
            max_fitness = genome.fitness
            max_fitness_genome = genome

    if be_elite:
        parents = [max_fitness_genome]
    else:
        parents = []

    while len(parents) < parent_count:
        threshold = random.random()
        index_of_smallest_probability_greater_than_threshold = -1
        for i in range(len(pop)):
            if probability_dict[i] > threshold:
                if (index_of_smallest_probability_greater_than_threshold == -1
                        or probability_dict[i] < probability_dict[index_of_smallest_probability_greater_than_threshold]):
                    index_of_smallest_probability_greater_than_threshold = i

        if index_of_smallest_probability_greater_than_threshold == -1:
            print("No parent index selected, you are bad at math")
            continue

        parents.append(pop[index_of_smallest_probability_greater_than_threshold])

    return parents


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # uncomment the below to make things run faster
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
