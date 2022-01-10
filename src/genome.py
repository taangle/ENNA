import time
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten

FITNESS_TIME_ALPHA = .0005


class Genome:
    def __init__(self, layers, batch_size, epochs, hidden_layer_count):
        # TODO give Genome objects GUIDs to keep track of ancestry
        self.layers = layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_layer_count = hidden_layer_count
        # a fitness and model of None indicate that fitness has not yet been determined for this genome
        self.fitness = None
        self.accuracy = None
        self.training_time = None
        self.model = None

    def calculate_fitness(self, data):
        (x_train, y_train), (x_test, y_test) = data

        self.model = Sequential()
        self._add_layers_to_model()
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        start_time = time.time()
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs)  # validation_split=0.1)
        end_time = time.time()
        self.training_time = end_time - start_time
        print("Training time:", self.training_time)

        score = self.model.evaluate(x_test, y_test, verbose=0)
        self.accuracy = score[1]
        # accounting for training time like this will hopefully just differentiate models that get similar accuracy
        #  with very different training times
        self.fitness = self.accuracy - (FITNESS_TIME_ALPHA * self.training_time)

    def _add_layers_to_model(self):
        for layer in self.layers:
            layer_object = None
            if layer["type"] == "Conv2D":
                layer_object = Conv2D(
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
        string = f"=Genome=\n\tFitness: {self.fitness}\n\tAccuracy: {self.accuracy}"\
            f"\n\tLayer count (all types): {len(self.layers)}"
        return string
