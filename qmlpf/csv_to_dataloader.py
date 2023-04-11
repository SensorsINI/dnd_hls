import numpy as np
import torch.utils.data
import csv
from sklearn.model_selection import train_test_split

class ds(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return(len(self.labels))

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return(feature, label)


def preprocess(fp, feature_col_start, feature_col_end, label_cols, test_size, batch_size):

    with open(fp) as f:
        reader = csv.reader(f)
        next(reader)
        dat = np.array(list(reader))

    features = dat[:,feature_col_start:feature_col_end]
    labels = dat[:,label_cols]

    features = [[float(i) for i in row] for row in features]
    labels = [float(j) for j in labels]

    split = train_test_split(features, labels, test_size=test_size)

    x_train, x_test, y_train, y_test = map(lambda x: torch.tensor(x), split)

    train_ds = ds(x_train, y_train)
    test_ds = ds(x_test, y_test)

    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    print('data loading done...')

    return trainloader, testloader
