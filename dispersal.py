import numpy as np, matplotlib.pyplot as plt
import pandas as pd, scipy.integrate as scint
import time, itertools,sys
import seaborn as sns
from competition import generate_prm,Path,mpfig,make_point_measure,make_measures



#### DEFINING PARAMETERS
# S=number of species, M=number of patches
parameters={'S':20,
            'M':41,
            'r':1,
            'Nmin': 10 ** -2,
            'tmax':1000,
            'mean':.5,
            'wid':.5,
            'sym':1}

ds = np.logspace(-2, 2, 4)
sigmas = np.linspace(1, 15, 4)

path=Path('graddisp')
plots=[
        'all', #PLOT ALL TRAJECTORIES IN A SINGLE MULTIPLOT (REMOVE IF MANY VALUES OF d AND sigma)
        'heatmap'
       ]

#### GENERATING MATRICES
RERUN = 'rerun' in sys.argv
if 'load' in sys.argv:
    #LOAD FROM FILE
    K = np.array(pd.read_csv(path+"Carr_Cap.csv", sep=',', header=None))
    A = np.array(pd.read_csv(path+"Competition.csv", sep=',', header=None))
    S, M = K.shape
    pos = np.linspace(100, 200, M)
    data={'alpha':A,}
else:
    #GENERATE NEW A AND K
    S,M=parameters['S'],parameters['M']
    data=generate_prm(**parameters)
    A, Kmax, lopt, tol = [data[z] for z in ('alpha', 'Kmax', 'lopt', 'tol')]
    np.fill_diagonal(A,1)
    pos=np.linspace(100,200,M)
    K =  np.atleast_2d(Kmax) * np.exp(-np.add.outer(pos, - lopt) ** 2 / (2 * np.atleast_2d(tol)** 2) )
    K=K.T

N0 = np.ones(K.shape) * 2
r,tmax,Nmin =[parameters[i] for i in ('r','tmax','Nmin')]
versions = [{'d': d, 'sigma': sigma} for d, sigma in itertools.product(ds, sigmas)]

#### RELOAD ALREADY COMPUTED RESULTS
path.mkdir()
results = None
dists = np.add.outer(pos, -pos)
if not RERUN:
    try:
        results = pd.read_json(path+'results.json')
    except:
        pass

table=[] #WILL CONTAIN STORED RESULTS
for vidx, version in enumerate(versions):
    tref = time.time()
    label='_'.join(['{}:{:.2g}'.format(i,version[i]) for i in sorted(version)])
    if not results is None and label in results['label'].values:
        row=results.set_index('label').loc[label ]
        row['label']=label
        table.append(row.to_dict() )
        continue
    print version
    d = version['d']
    sigma = version['sigma']
    lap = np.exp(- dists.astype('float') ** 2 / (2 *(10**-4+ sigma) ** 2))
    #lap[sigma==0]=0
    np.fill_diagonal(lap, 0)
    lap = lap - np.diag(np.sum(lap, axis=1))

    lap = lap.reshape(M, M)
    dmat = d * lap
    lastt = list(np.logspace(-4, 4, 100))

    def eqs(t, N):
        N = np.clip(N.reshape(K.shape), 0, None)

        dN = (r * N * (1 - np.dot(A, N)/K ) + np.dot(N, dmat))
        dN[N < Nmin] = np.clip(dN[N < Nmin], 0, None)
        if lastt and t > lastt[0]:
            lastt.pop(0)
            print t, np.min(N), np.max(N), np.min(dN), N.ravel()[np.argmin(dN)], np.argmin(dN)
        return dN.ravel()


    integrator = scint.ode(eqs).set_integrator('dop853', nsteps=100000)

    integrator.set_initial_value(N0.ravel(), 0)
    result = integrator.integrate(tmax).reshape(K.shape)

    #MAKE MEASUREMENTS (STORE RESULTS)
    dic={'label':label,'Nf':result,'runtime': time.time() - tref}
    dic.update(parameters)
    dic.update(version)

    #MEASUREMENTS AT EACH POINT OF THE GRADIENT
    df=[]
    dNdt=eqs(tmax,result).reshape(K.shape) #derivatives
    for i, p in enumerate(pos):
        measure,locdata={'position':p, 'Nf':result[:,i],'dNdt':dNdt[:,i] },{'selfint':np.ones(S),'community':A,'growth':K[:,i]}
        make_point_measure(locdata, measure,death=Nmin)
        df.append(measure)
    #COMBINED MEASUREMENTS OVER THE WHOLE GRADIENT
    make_measures(pd.DataFrame(df),dic,death=Nmin)
    table.append(dic)

results = pd.DataFrame(table)
results.to_json(path+'results.json')

results.reset_index()
#### PLOTS
if 'all' in plots:
    npanels=len(results)
    side=np.ceil(np.sqrt(npanels)).astype('int')
    plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize'])/2.*(side,side))
    for vidx, row in results.iterrows():
        result = np.array(row['Nf'])
        plt.subplot(side, side, vidx + 1)
        plt.plot(result.T)
        plt.title(','.join(['{}={}'.format(i, row[i]) for i in ['d', 'sigma']]) + '\nruntime={:.2f}'.format(
            row['runtime']))
if 'heatmap' in plots:
    df=results
    df['stdJ'] = [np.std(x) for x in [np.array(z) for z in df['Jaccard'].values]]
    for val in ['stdJ','alive']:
        showdf=df[['d','sigma',val]]
        tab=showdf.pivot(columns='d', index='sigma', values=val)
        vmax = np.max(np.abs(tab.values))
        plt.figure()
        if np.min(tab.values)<0 and np.max(tab.values)>0:
            sns.heatmap(tab, vmax=vmax, vmin=-vmax, cmap='seismic_r'),plt.title(val)
        else:
            sns.heatmap(tab,cmap='PuRd'),plt.title(val)
        plt.gca().invert_yaxis()
        plt.savefig(path + '{}.pdf'.format(val))
plt.show()


