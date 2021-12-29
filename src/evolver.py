import copy
import random

from src.genome import Genome
from src.histories import record_generation

PRELIMINARY_LAYER_COUNT = 2
POSTLIMINARY_LAYER_COUNT = 1
MIN_HIDDEN_MUTATIONS = 1
MAX_HIDDEN_MUTATIONS = 6
UNIT_MUTATION_MARGIN = 4
DROPOUT_APPEARANCE_CHANCE = .2
HIDDEN_LAYER_COUNT_MUTATION_CHANCE = .25
ACTIVATION_MUTATION_CHANCE = .2
FILTER_MUTATION_CHANCE = .2
KERNEL_MUTATION_CHANCE = .2
STRIDE_MUTATION_CHANCE = .2


class Evolver:
    def __init__(self,
                 data, start_time_str,
                 pop_size, max_gen, gen_to_start_being_elite, parent_count, batch_size, epochs, input_shape,
                 num_classes, min_filters, max_filters, min_kernel_dim, max_kernel_dim, min_stride_dim, max_stride_dim,
                 min_hidden_layers, max_hidden_layers, min_units, max_units, max_dropout):
        self.data = data
        self.start_time_str = start_time_str
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.gen_to_start_being_elite = gen_to_start_being_elite
        self.parent_count = parent_count
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.min_filters = min_filters
        self.max_filters = max_filters
        self.min_kernel_dim = min_kernel_dim
        self.max_kernel_dim = max_kernel_dim
        self.min_stride_dim = min_stride_dim
        self.max_stride_dim = max_stride_dim
        self.min_hidden_layers = min_hidden_layers
        self.max_hidden_layers = max_hidden_layers
        self.min_units = min_units
        self.max_units = max_units
        self.max_dropout = max_dropout

    def generate_random_population(self):
        population = []
        for genome_index in range(self.pop_size):
            layers = []
            kernel_dim = random.randint(self.min_kernel_dim, self.max_kernel_dim)
            stride_dim = random.randint(self.min_stride_dim, self.max_stride_dim)
            layers.append({
                "type": 'Conv2D',
                "input_shape": self.input_shape,
                "filters": random.randint(self.min_filters, self.max_filters),
                "kernel_size": (kernel_dim, kernel_dim),
                "strides": (stride_dim, stride_dim),
                "activation": pick_random_activation(),
            })
            # TODO maxpool? more Conv2d?
            layers.append({"type": 'Flatten'})

            hidden_layer_count = random.randint(self.min_hidden_layers, self.max_hidden_layers)
            self._append_dense_and_dropout_layers(layers, hidden_layer_count)

            layers.append(get_an_output_layer(self.num_classes))

            population.append(Genome(layers, self.batch_size, self.epochs, hidden_layer_count))

        # test fitness (opportunity for parallelization)
        for genome in population:
            genome.calculate_fitness(self.data)
            print(genome)

        record_generation(0, population, self.start_time_str)
        print("==End of generation 0==")

        return population

    def evolve(self, population):
        pop = population
        for gen_number in range(self.max_gen):
            # select parents
            its_time_to_be_elitist = gen_number >= self.gen_to_start_being_elite
            parents = self.select_parents(pop, be_elitist=its_time_to_be_elitist)

            # refill population with mutated children of parents
            pop = []
            while len(pop) < self.pop_size:
                random.shuffle(parents)
                # if PARENT_COUNT is odd, one parent will get left out, but not necessarily indefinitely
                pairs = zip(parents[::2], parents[1::2])
                for genome0, genome1 in pairs:
                    pop.append(self.combine_genomes(genome0, genome1))
                    if len(pop) == self.pop_size:
                        break

            # test fitness
            for genome in pop:
                if genome.fitness is None:
                    genome.calculate_fitness(self.data)

                print(genome)

            # TODO it's dumb that the generation number should be one off like this
            record_generation(gen_number + 1, pop, self.start_time_str)
            print(f"==end of generation {gen_number + 1}==")

        return pop

    # TODO switch everything to bit crossover?
    def combine_genomes(self, first_genome, second_genome):
        first_layer = random.choice([copy.deepcopy(first_genome.layers[0]), copy.deepcopy(second_genome.layers[0])])
        if random.random() < FILTER_MUTATION_CHANCE:
            first_layer["filters"] = random.randint(self.min_filters, self.max_filters)
        if random.random() < KERNEL_MUTATION_CHANCE:
            kernel_dim = random.randint(self.min_kernel_dim, self.max_kernel_dim)
            first_layer["kernel_size"] = (kernel_dim, kernel_dim)
        if random.random() < STRIDE_MUTATION_CHANCE:
            stride_dim = random.randint(self.min_stride_dim, self.max_stride_dim)
            first_layer["strides"] = (stride_dim, stride_dim)

        layers = [first_layer, {"type": "Flatten"}]
        extra_hidden_layers = []
        hidden_layer_count = random.choice([first_genome.hidden_layer_count, second_genome.hidden_layer_count])
        if random.random() < HIDDEN_LAYER_COUNT_MUTATION_CHANCE:
            hidden_layer_count = min(self.max_hidden_layers,
                                     max(self.min_hidden_layers, hidden_layer_count + random.randint(-2, 2)))
            # layer_diff is the number of layers *more* that exist in the child than in the parent with the most layers
            # in other words, the number of layers that we need to create from scratch
            layer_diff = hidden_layer_count - max(first_genome.hidden_layer_count, second_genome.hidden_layer_count)
            if layer_diff > 0:
                self._append_dense_and_dropout_layers(extra_hidden_layers, layer_diff)

        # TODO clean this section up, double-check its logic
        inner_layer_count0 = len(first_genome.layers) - PRELIMINARY_LAYER_COUNT - POSTLIMINARY_LAYER_COUNT
        inner_layer_count1 = len(second_genome.layers) - PRELIMINARY_LAYER_COUNT - POSTLIMINARY_LAYER_COUNT
        min_original_inner_layer_count = min(inner_layer_count0, inner_layer_count1)
        max_original_inner_layer_count = max(inner_layer_count0, inner_layer_count1)
        for i in range(max_original_inner_layer_count):
            inner_index = i + PRELIMINARY_LAYER_COUNT
            # if the abstract index i has not yet exceeded the point where both genomes are available, pick from one
            if i < min_original_inner_layer_count:
                layers.append(random.choice([
                    copy.deepcopy(first_genome.layers[inner_index]),
                    copy.deepcopy(second_genome.layers[inner_index])]))
            # else one of the genomes is exhausted, if it is genome1 we use genome0
            elif i < inner_layer_count0:
                layers.append(copy.deepcopy(first_genome.layers[inner_index]))
            # if it is genome0 we use genome1
            elif i < inner_layer_count1:
                layers.append(copy.deepcopy(second_genome.layers[inner_index]))

        layers = layers + extra_hidden_layers

        number_of_hidden_changes = random.randint(MIN_HIDDEN_MUTATIONS, MAX_HIDDEN_MUTATIONS)
        for i in range(number_of_hidden_changes):
            layer_to_change = 0
            while layers[layer_to_change]["type"] != "Dense":
                layer_to_change = random.randint(PRELIMINARY_LAYER_COUNT, len(layers) - 1)
            units = layers[layer_to_change]["units"]
            layers[layer_to_change]["units"] = min(self.max_units, max(self.min_units,
                                                                  random.randint(units - UNIT_MUTATION_MARGIN,
                                                                                 units + UNIT_MUTATION_MARGIN)))
            if random.random() < ACTIVATION_MUTATION_CHANCE:
                layers[layer_to_change]["activation"] = pick_random_activation()

        layers.append(get_an_output_layer(self.num_classes))

        # TODO if learning batch_size and epochs, update this
        combined = Genome(layers, first_genome.batch_size, first_genome.epochs, hidden_layer_count)
        return combined

    def select_parents(self, population, be_elitist=False):
        total_fitness = sum(genome.fitness for genome in population)
        probability_dict = {}
        previous_prob = 0
        max_fitness = 0
        max_fitness_genome = None
        for i in range(len(population)):
            genome = population[i]
            probability_dict[i] = previous_prob + (genome.fitness / total_fitness)
            previous_prob = probability_dict[i]
            if genome.fitness > max_fitness:
                max_fitness = genome.fitness
                max_fitness_genome = genome

        if be_elitist:
            parents = [max_fitness_genome]
        else:
            parents = []

        while len(parents) < self.parent_count:
            threshold = random.random()
            index_of_smallest_probability_greater_than_threshold = -1
            for i in range(len(population)):
                if probability_dict[i] > threshold:
                    if (index_of_smallest_probability_greater_than_threshold == -1
                            or probability_dict[i] < probability_dict[
                                index_of_smallest_probability_greater_than_threshold]):
                        index_of_smallest_probability_greater_than_threshold = i

            if index_of_smallest_probability_greater_than_threshold == -1:
                print("No parent index selected, you are bad at math")
                continue

            parents.append(population[index_of_smallest_probability_greater_than_threshold])

        return parents

    def _append_dense_and_dropout_layers(self, layers, count):
        for dense_layer_index in range(count):
            layers.append({
                "type": 'Dense',
                "units": random.randint(self.min_units, self.max_units),
                "activation": pick_random_activation()
            })
            if random.random() < DROPOUT_APPEARANCE_CHANCE:
                layers.append({
                    "type": 'Dropout',
                    "rate": random.uniform(0, self.max_dropout)
                })


# TODO test these out
def pick_random_activation():
    return random.choice(['relu'])  # , 'elu', 'tanh', 'sigmoid'])


def get_an_output_layer(num_classes):
    return {
        "type": "Dense",
        "units": num_classes,
        "activation": "softmax"
    }
