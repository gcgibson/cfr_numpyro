

import os
import numpy as np
import covid.models.SEIRD

path ="/mnt/nfs/work1/sheldon/gcgibson/llonger_H/2020-05-17/"


def load_samples(filename):

    x = np.load(filename, allow_pickle=True)

    prior_samples = x['prior_samples'].item()
    mcmc_samples = x['mcmc_samples'].item()
    post_pred_samples = x['post_pred_samples'].item()
    forecast_samples = x['forecast_samples'].item()

    return prior_samples, mcmc_samples, post_pred_samples, forecast_samples

## read the samples object

prior_samples, mcmc_samples, post_pred_samples, forecast_samples = \
        load_samples(path+"/samples/AK.npz")

model  =covid.models.SEIRD.SEIRD_incident

forecast_field='dz'

z = model.get(forecast_samples, forecast_field, forecast=True)
print (z)

