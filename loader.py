
import numpy as np
from scipy import stats
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data



# generate 1000 samples from the bivariate normal distribution
def generate_normal(
    mean: list[float],
    cov: list[list[float]],
    size: int
):
    return np.random.multivariate_normal(mean, cov, size=size)


def generate_normal_bernoulliASmean(
    mean: list[float],
    cov: list[list[float]],
    size: int,
    binomial_maen: list[float],
    p_success: float
):

    bernoulli_sample = stats.bernoulli.rvs(p=p_success, size=size)[:, np.newaxis]
    binomial_maen_tile = np.tile(binomial_maen, (size,1))
    ret = generate_normal(mean, cov, size) + bernoulli_sample * binomial_maen_tile
    return  ret


data_list = [Data(...), ..., Data(...)]
loader = DataLoader(data_list, batch_size=32)


def make_dataset_2d_each(
    mean,
    cov,
    size,
    noise_cov,
    binomial_maen,
    p_success,
    p_noise=0.2
):
    features = stats.bernoulli.rvs(p=p_noise, size=size)
    noise = generate_normal(
        [0, 0],
        noise_cov,
        size
    )
    # normal = generate_normal(mean, cov, size)
    data = generate_normal_bernoulliASmean(
        mean,
        cov,
        size,
        binomial_maen,
        p_success
    )

    data_noise = data + features[:, np.newaxis] * noise
    # ret = [ Data(x=f, pos=d) for d, f in zip(data_noise, features)]
    
    return data_noise, features
        
def make_dataset(
    meanList,
    covList,
    size,
    noise_cov,
    binomial_maen,
    p_success,
    p_noise=0.2,
    y=0
):  
    dataList = list()
    featuresList = list()
    
    for mean, cov in zip(meanList,covList):
        data, features = make_dataset_2d_each(
            mean,
            cov,
            size,
            noise_cov,
            binomial_maen,
            p_success,
            p_noise
        )
        print(data.shape)
        dataList.append(data)
        featuresList.append(features)

    # print(np.array(dataList).shape, np.array(featuresList).shape)
    dataList = np.hstack(dataList)
    featuresList = np.array(featuresList).T
    # print(dataList.shape, featuresList.shape)

    print(dataList[0])
    ret = [Data(x=f, pos=d, y=y) for d, f in zip(dataList, featuresList)]
    print(ret[0].x)

    return ret


def generation():

    mean = [[0, 0], [1, 1], [2, 2]]
    cov = [
        [
            [0.2, 0],
            [0, 0.2]
        ],
        [
            [0.2, 0],
            [0, 0.2]
        ],
        [
            [0.2, 0],
            [0, 0.2]
        ]
    ]
    noise_cov1 = [
        [1, 0],
        [0, 1]
    ]
    size = 2000
    binomial_maen = [3, 3]
    p_success = 0.1
    p_noise = 0.2
    dataset1 = make_dataset(
        mean,
        cov,
        size,
        noise_cov1,
        binomial_maen,
        p_success,
        p_noise,
        0
    )

    mean2 = [[1, 0], [1, 1], [2, 2]]
    cov2 = [
        [
            [0.2, 0],
            [0, 0.2]
        ],
        [
            [0.2, 0],
            [0, 0.2]
        ],
        [
            [0.2, 0],
            [0, 0.2]
        ]
    ]
    size2 = 2000
    noise_cov2 = [
        [1, 0],
        [0, 1]
    ]
    size = 2000
    binomial_maen2 = [3, 3]
    p_success2 = 0.1
    p_noise2 = 0.2
    dataset2 = make_dataset(
        mean2,
        cov2,
        size2,
        noise_cov2,
        binomial_maen2,
        p_success2,
        p_noise2,
        1
    )
    print(dataset1[0])
    print(dataset2[0])

if __name__ == '__main__':

    generation()