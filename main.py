from mlts.models import LSTM


def run():
    my_model = LSTM()
    my_model.fit()
    my_model.predict()


if __name__ == '__main__':
    run()
