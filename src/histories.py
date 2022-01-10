import csv
from pathlib import Path

HISTORY_DIR = "../histories"
INFO_FILE_NAME = "info.csv"
HISTORY_FILE_NAME = "history.csv"


# TODO just use a DB instead
def initialize_history_dir(evolver):
    Path(f"{HISTORY_DIR}/{evolver.start_time_str}").mkdir(parents=True, exist_ok=True)

    with open(f"{HISTORY_DIR}/{evolver.start_time_str}/{INFO_FILE_NAME}", mode="w", newline="") as info_file:
        info_writer = csv.writer(info_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        # TODO add the rest of the params
        info_writer.writerow(
            ["batch size", "epochs", "pop_size", "max_gen", "gen_to_start_being_elite", "parent_count", "min units",
             "max units"])
        info_writer.writerow(
            [evolver.batch_size,
             evolver.epochs,
             evolver.pop_size,
             evolver.max_gen,
             evolver.gen_to_start_being_elite,
             evolver.parent_count,
             evolver.min_units,
             evolver.max_units]
        )

    with open(f"{HISTORY_DIR}/{evolver.start_time_str}/{HISTORY_FILE_NAME}", mode="w", newline="") as history_file:
        history_writer = csv.writer(history_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        history_writer.writerow(
            ["generation", "genome", "accuracy", "training_time", "fitness", "layer", "type", "units", "activation"]
        )


def record_generation(gen_number, population, start_time_str):
    with open(f"{HISTORY_DIR}/{start_time_str}/{HISTORY_FILE_NAME}", mode="a", newline="") as history_file:
        history_writer = csv.writer(history_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for genome_index in range(len(population)):
            genome = population[genome_index]
            for layer_index in range(len(genome.layers)):
                layer = genome.layers[layer_index]
                # TODO add the other layers
                if layer["type"] == "Dense":
                    history_writer.writerow([
                        gen_number,
                        genome_index,
                        genome.accuracy,
                        genome.training_time,
                        genome.fitness,
                        layer_index,
                        layer["type"],
                        layer["units"],
                        layer["activation"]
                    ])
