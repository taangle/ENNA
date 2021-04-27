import json
import random
import copy
import time
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

PRELIMINARY_LAYER_COUNT = 2
POSTLIMINARY_LAYER_COUNT = 1
MIN_HIDDEN_MUTATIONS = 1
MAX_HIDDEN_MUTATIONS = 6
UNIT_MUTATION_MARGIN = 32
HIDDEN_LAYER_COUNT_MUTATION_CHANCE = .2
ACTIVATION_MUTATION_CHANCE = .2
FILTER_MUTATION_CHANCE = .2
KERNEL_MUTATION_CHANCE = .2
STRIDE_MUTATION_CHANCE = .2


class Genome:
    def __init__(self, layers, batch_size, epochs, hidden_layer_count):
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_layer_count = hidden_layer_count
        # a fitness and model of None indicate that fitness has not yet been determined for this genome
        self.fitness = None
        self.accuracy = None
        self.model = None

    def calculate_fitness(self, data):
        (x_train, y_train), (x_test, y_test) = data

        self.model = Sequential()
        self._add_layers_to_model()
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        start_time = time.time()
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)  # validation_split=0.1)
        end_time = time.time()
        training_time = end_time - start_time
        print("Training time:", training_time)

        score = self.model.evaluate(x_test, y_test, verbose=0)
        self.accuracy = score[1]
        # TODO adjust how training time affects fitness, currently is negligible for long training times
        self.fitness = self.accuracy + (1 / training_time)

    def _add_layers_to_model(self):
        for layer in self.layers:
            layer_object = None
            if layer["type"] == "Conv2D":
                layer_object = Conv2D(
                    # TODO if you add another Conv2D, it won't have input_shape
                    input_shape=layer["input_shape"],
                    filters=layer["filters"],
                    kernel_size=layer["kernel_size"],
                    strides=layer["strides"],
                    activation=layer["activation"],
                    # TODO think about this/test this
                    padding='valid'
                )
            elif layer["type"] == "Flatten":
                layer_object = Flatten()
            elif layer["type"] == "Dense":
                layer_object = Dense(units=layer["units"], activation=layer["activation"])
            elif layer["type"] == "Dropout":
                layer_object = Dropout(rate=layer["rate"])

            self.model.add(layer_object)

    def __str__(self):
        string = f"=Genome=\n\tFitness: {self.fitness}\n\tAccuracy: {self.accuracy}\n\tLayers:"
        for layer in self.layers:
            string += f"\n\t\t{json.dumps(layer)}"

        return string


def pick_random_activation():
    return random.choice(['relu', 'elu', 'tanh', 'sigmoid'])


def generate_random_population(pop_size, batch_size, epochs, input_shape, min_filters, max_filters, min_kernel_dim, max_kernel_dim, min_stride_dim, max_stride_dim, min_hidden_layers, max_hidden_layers, min_units, max_units, max_dropout, num_classes):
    pop = []
    for genome_index in range(pop_size):
        layers = []
        kernel_dim = random.randint(min_kernel_dim, max_kernel_dim)
        stride_dim = random.randint(min_stride_dim, max_stride_dim)
        layers.append({
            "type": 'Conv2D',
            "input_shape": input_shape,
            "filters": random.randint(min_filters, max_filters),
            "kernel_size": (kernel_dim, kernel_dim),
            "strides": (stride_dim, stride_dim),
            "activation": pick_random_activation(),
        })
        # TODO maxpool? more Conv2d?
        layers.append({"type": 'Flatten'})

        hidden_layer_count = random.randint(min_hidden_layers, max_hidden_layers)
        for dense_layer_index in range(hidden_layer_count):
            layers.append({
                "type": 'Dense',
                "units": random.randint(min_units, max_units),
                "activation": pick_random_activation()
            })
            layers.append({
                "type": 'Dropout',
                "rate": random.uniform(0, max_dropout)
            })

        layers.append(get_an_output_layer(num_classes))

        pop.append(Genome(layers, batch_size, epochs, hidden_layer_count))

    return pop


def get_an_output_layer(num_classes):
    return {
        "type": "Dense",
        "units": num_classes,
        "activation": "softmax"
    }


# TODO switch everything to bit crossover?
def combine_genomes(genome0, genome1, min_filters, max_filters, min_kernel_dim, max_kernel_dim, min_stride_dim, max_stride_dim, min_hidden_layers, max_hidden_layers, min_units, max_units, max_dropout, num_classes):
    first_layer = random.choice([copy.deepcopy(genome0.layers[0]), copy.deepcopy(genome1.layers[0])])
    if random.random() < FILTER_MUTATION_CHANCE:
        first_layer["filters"] = random.randint(min_filters, max_filters)
    if random.random() < KERNEL_MUTATION_CHANCE:
        kernel_dim = random.randint(min_kernel_dim, max_kernel_dim)
        first_layer["kernel_size"] = (kernel_dim, kernel_dim)
    if random.random() < STRIDE_MUTATION_CHANCE:
        stride_dim = random.randint(min_stride_dim, max_stride_dim)
        first_layer["strides"] = (stride_dim, stride_dim)

    layers = [first_layer, {"type": "Flatten"}]
    extra_hidden_layers = []
    hidden_layer_count = random.choice([genome0.hidden_layer_count, genome1.hidden_layer_count])
    if random.random() < HIDDEN_LAYER_COUNT_MUTATION_CHANCE:
        hidden_layer_count = min(max_hidden_layers, max(min_hidden_layers, hidden_layer_count + random.randint(-2, 2)))
        # the following is the number of layers *more* that exist in the child than in the parent with the most layers
        # in other words, the number of layers that we need to create from scratch
        layer_diff = hidden_layer_count - max(genome0.hidden_layer_count, genome1.hidden_layer_count)
        if layer_diff > 0:
            for i in range(layer_diff):
                extra_hidden_layers.append({
                    "type": 'Dense',
                    "units": random.randint(min_units, max_units),
                    "activation": pick_random_activation()
                })
                extra_hidden_layers.append({
                    "type": 'Dropout',
                    "rate": random.uniform(0, max_dropout)
                })

    # I know this section is a little convoluted, but I have double-checked it and I think it's correct
    inner_layer_count0 = len(genome0.layers) - PRELIMINARY_LAYER_COUNT - POSTLIMINARY_LAYER_COUNT
    inner_layer_count1 = len(genome1.layers) - PRELIMINARY_LAYER_COUNT - POSTLIMINARY_LAYER_COUNT
    min_original_inner_layer_count = min(inner_layer_count0, inner_layer_count1)
    max_original_inner_layer_count = max(inner_layer_count0, inner_layer_count1)
    for i in range(max_original_inner_layer_count):
        inner_index = i + PRELIMINARY_LAYER_COUNT
        # if the abstract index i has not yet exceeded the point where both genomes are available, pick from one
        if i < min_original_inner_layer_count:
            layers.append(random.choice([
                copy.deepcopy(genome0.layers[inner_index]),
                copy.deepcopy(genome1.layers[inner_index])]))
        # else one of the genomes is exhausted, if it is genome1 we use genome0
        elif i < inner_layer_count0:
            layers.append(copy.deepcopy(genome0.layers[inner_index]))
        # if it is genome0 we use genome1
        elif i < inner_layer_count1:
            layers.append(copy.deepcopy(genome1.layers[inner_index]))

    layers = layers + extra_hidden_layers

    number_of_hidden_changes = random.randint(MIN_HIDDEN_MUTATIONS, MAX_HIDDEN_MUTATIONS)
    for i in range(number_of_hidden_changes):
        layer_to_change = 0
        while layers[layer_to_change]["type"] != "Dense":
            layer_to_change = random.randint(PRELIMINARY_LAYER_COUNT, len(layers) - 1)
        units = layers[layer_to_change]["units"]
        layers[layer_to_change]["units"] = max(min_units, random.randint(units - UNIT_MUTATION_MARGIN, units + UNIT_MUTATION_MARGIN))
        if random.random() < ACTIVATION_MUTATION_CHANCE:
            layers[layer_to_change]["activation"] = pick_random_activation()

    layers.append(get_an_output_layer(num_classes))

    # TODO if learning batch_size and epochs, update this
    combined = Genome(layers, genome0.batch_size, genome0.epochs, hidden_layer_count)
    return combined
