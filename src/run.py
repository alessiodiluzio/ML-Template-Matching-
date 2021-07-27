from src.training import train
from src.dataset import get_dataset
from src import EPOCHS


def run_train():
    training_set, validation_set, train_step, val_step= get_dataset(show=False)
    train(training_set, validation_set, EPOCHS, train_step, val_step, plot_path='plot', image_path='image', early_stopping=15)


# TODO
def run_test():
    pass
