from . import dataset


class Ridge(BasicPair):
    Data = dataset.sk_data
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = self.Data(self.dataset)
    
    def compile(self, **kwargs):
        from sklearn.linear_model import Ridge as Model
        self.model = Model(**kwargs)
    
    def fit(self):
        x, y = self.data.get_xy('train')
        model.fit(x, y)
        # vali = self.evaluate('vali')
        # msg = 'vali: {}, time: {:.1f}s {:.1f}s'.format(vali, train_time, vali_time)
    
    def predict(self, x):
        return self.model.predict(x)


def main():
    print('hello world, sklean_models.py')


if __name__ == '__main__':
    main()
