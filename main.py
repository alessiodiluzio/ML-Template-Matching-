import sys
import tensorflow as tf

from src.run import run_train, run_test

message = 'Usage: python main.py <mode> [-p <data_path>]\n' \
          'mode:\n' \
          '\ttrain to run training.\n' \
          '\ttest to run prediction.\n' \
          '-p train/test performed on images contained at data_path.\n'


def main(_):
    arg = sys.argv[1:]
    if len(arg) < 1:
        print(message)
    elif arg[0].lower() == 'train':
        run_train()
    elif arg[0].lower() == 'test':
        run_test()
    else:
        print(message)


if __name__ == "__main__":
    tf.compat.v1.app.run()




