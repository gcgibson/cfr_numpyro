import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

import pandas as pd
import numpy as np
DATA_URI = "https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-truth/truth-Incident%20Cases.csv"

df = pd.read_csv(DATA_URI)

DATA_URI = "https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-truth/truth-Incident%20Deaths.csv"


df_deaths = pd.read_csv(DATA_URI)

fips_uri = "https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-locations/locations.csv"

df_fips = pd.read_csv(fips_uri)

cases = df.set_index('location_name').join(df_fips.set_index('location_name'),how='left',lsuffix='test')
deaths = df_deaths.set_index('location_name').join(df_fips.set_index('location_name'),how='left',lsuffix='test')


region = sys.argv[1]#"Florida"
date = sys.argv[2]#"2020-11-08"


cases=cases[cases.abbreviation==region]
deaths=deaths[deaths.abbreviation==region]


print (cases)


joined_df_full = cases.set_index('date').join(deaths.set_index('date'),how='left',lsuffix='cases',rsuffix='deaths')
joined_df = joined_df_full.loc[joined_df_full.index <= date]

joined_df_full.index = pd.to_datetime(joined_df_full.index)
joined_test = joined_df_full.loc[(joined_df_full.index >= pd.to_datetime(date) +pd.Timedelta("1 days")) & (joined_df_full.index <= pd.to_datetime(date) +pd.Timedelta("28 days"))]


joined_test_full = joined_df_full.loc[ (joined_df_full.index <= pd.to_datetime(date) +pd.Timedelta("28 days"))]





import rpy2
import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
splines = importr("splines")
predict = importr("stats")


df_full = pd.DataFrame({'X':np.arange(len(joined_df.valuecases.tolist())),
                   'Y':np.array(joined_df.valuecases.tolist()),  # add your data here
                    'D': np.array(joined_df.valuedeaths.tolist())})
                  
df_train =df_full
basis = splines.ns(np.arange(0,len(df_full.X.values)),knots=np.arange(0,len(df_full.X.values),20))
basis_matrix = np.array(basis)
basis_train= basis_matrix[:len(df_full.X.values),:]

basis_oos = predict.predict(basis,newx=np.arange(1,(len(df_full.X.values)+29)))
basis_oos_matrix = np.array(basis_oos)
num_data =basis_matrix.shape[0]
num_basis = basis_matrix.shape[1]





def ExponentialRandomWalk(loc=1., scale=1e-2, drift=0., num_steps=100):
    '''
    Return distrubtion of exponentiated Gaussian random walk
    
    Variables are x_0, ..., x_{T-1}
    
    Dynamics in log-space are random walk with drift:
       log(x_0) := log(loc) 
       log(x_t) := log(x_{t-1}) + drift + eps_t,    eps_t ~ N(0, scale)
        
    ==> Dynamics in non-log space are:
        x_0 := loc
        x_t := x_{t-1} * exp(drift + eps_t),    eps_t ~ N(0, scale)        
    '''
    
    log_loc = np.log(loc) + drift * (np.arange(num_steps)+0.)
    
    return dist.TransformedDistribution(
        dist.GaussianRandomWalk(scale=scale, num_steps=num_steps),
        [
            dist.transforms.AffineTransform(loc = log_loc, scale=1.),
            dist.transforms.ExpTransform()
        ]
    )





mask = np.zeros((num_data,num_data))

for i in range(num_data):
    mask[i,:(i+1)] = 1


case_normalizer = np.max(df_train.Y.values)
df_train.Y = df_train.Y.values/case_normalizer

death_normalizer = np.max(df_train.D.values)
df_train.D = df_train.D.values/death_normalizer



mask_pred = np.zeros((num_data+28,num_data+28))

for i in range(num_data+28):
    mask_pred[i,:(i+1)] = 1



def model(B_local=None,Forecast=False,mask=None,beta_1_tmp=None):
   
    a_raw = numpyro.sample('a_raw_init',ExponentialRandomWalk(scale=100, num_steps=num_basis))
    beta_1 = numpyro.sample('transform',dist.Normal(jnp.zeros(1),jnp.ones(1)))
    #det_prob = numpyro.sample('det_prob',LogisticRandomWalk(loc=.1,scale=1000,num_steps=num_data))
    if not Forecast:
        beta_1 = numpyro.sample('beta_1',dist.GaussianRandomWalk(scale=1e-3, num_steps=num_data))
    else:
        beta_1=beta_1_tmp
    
    y_hat =  numpyro.deterministic('y_hat', jnp.dot(jnp.array(B_local), a_raw))

    numpyro.sample('c_obs', dist.Normal(y_hat, 1e-2), obs=df_train.Y.values)
    #tiled_beta = jnp.tile(beta_1,(num_data,num_data))
    
    tmp = mask*beta_1[:,None]
    d_hat = numpyro.deterministic('d_hat',jnp.dot(tmp,y_hat))
    numpyro.sample('d_obs', dist.Normal(d_hat, 1e-1), obs=df_train.D.values)

    




rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

num_warmup, num_samples = 1000, 2000
# Run NUTS.
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup, num_samples)
mcmc.run(rng_key_,B_local=basis_train,mask=mask)
mcmc.print_summary()
samples_1 = mcmc.get_samples()



from numpyro.infer import Predictive
tmp = np.concatenate((samples_1['beta_1'].mean(axis=0),np.repeat(samples_1['beta_1'].mean(axis=0)[-1],28)))
predictive = Predictive(model, samples_1)
predictions = predictive(rng_key_,
                         Forecast=True,B_local=basis_oos,beta_1_tmp=tmp,mask=mask_pred)['d_hat']




forecasts = (np.median(predictions,0)*death_normalizer)[-28:]
truth = joined_test.valuedeaths.values

one_week_ahead_mae = np.abs( np.sum(forecasts[0:7])-np.sum(truth[0:7]))
two_week_ahead_mae = np.abs( np.sum(forecasts[7:14])-np.sum(truth[7:14]))
three_week_ahead_mae = np.abs( np.sum(forecasts[14:21])-np.sum(truth[14:21]))
four_week_ahead_mae = np.abs( np.sum(forecasts[21:28])-np.sum(truth[21:28]))

output_dir = "/mnt/nfs/work1/sheldon/gcgibson/output/"

mae = np.array([one_week_ahead_mae,two_week_ahead_mae,three_week_ahead_mae,four_week_ahead_mae])
np.savetxt(output_dir +region+"_"+date+"_"+".csv", mae, delimiter=",")



plt.style.use('ggplot')

fig1, ax1 = plt.subplots()



ax1.plot(df_train.X.values,samples_1['y_hat'].mean(axis=0),linewidth=.5)
ax1.scatter(df_train.X.values,df_train.Y.values,color='black',s=.75)

filename= "/mnt/nfs/work1/sheldon/gcgibson/figs/"+region+"_"+date+"_"+"cases.png"
fig1.savefig(filename, dpi=300)
plt.close()






fig2, ax2 = plt.subplots()



ax2.plot(df_train.X.values,samples_1['d_hat'].mean(axis=0),linewidth=.5)
ax2.scatter(df_train.X.values,df_train.D.values,color='black',s=.75)

filename= "/mnt/nfs/work1/sheldon/gcgibson/figs/"+region+"_"+date+"_"+"deaths.png"
fig2.savefig(filename, dpi=300)
plt.close()





fig3, ax3 = plt.subplots()



ax3.plot((np.median(predictions,0)*death_normalizer))
ax3.scatter(np.arange(len(joined_test_full.valuedeaths.values)),joined_test_full.valuedeaths.values,color='black',s=.75)

filename= "/mnt/nfs/work1/sheldon/gcgibson/figs/"+region+"_"+date+"_"+"preds.png"
fig3.savefig(filename, dpi=300)
plt.close()



