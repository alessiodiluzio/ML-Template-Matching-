import sys
import tensorflow as tf

from src.run import run_train, run_test
from src import OS

message = 'Usage: python main.py <mode> [-p <data_path>]\n' \
          'mode:\n' \
          '\ttrain to run training.\n' \
          '\ttest to run prediction.\n' \
          '-p train/test performed on images contained at data_path.\n'


def main():
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
    if OS == 'Darwin':
        from tensorflow.python.compiler.mlcompute import mlcompute
        mlcompute.set_mlc_device('gpu')
        print('Train on M1 GPU')
    print(f'Running on platform: {OS}')
    main()




