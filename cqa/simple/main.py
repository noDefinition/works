# import models_prev
from simple import models


# import dataset

def main():
    # model = models_prev.test('test')
    # Model = models_prev.Ans
    Model = models.Features
    # dataset = 'test'
    dataset = 'so'
    gpu = 0
    msg = ('{}/{}/gpu={}'.format(Model.__name__, dataset, gpu))
    print(msg)
    model = Model(dataset=dataset, batch_size=32, lr=1e-5, gpu=gpu)
    model.compile()
    model.fit()
    e = model.evaluate()
    msg = ('{}/{}/{}: {}'.format(model.run_name, Model.__name__, dataset, e))
    print(msg)
    if dataset != 'test':
        with open('log.txt', 'a') as f:
            f.write(msg + '\n')
    model.terminate()


if __name__ == '__main__':
    main()
