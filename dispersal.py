import numpy as np, matplotlib.pyplot as plt
import pandas as pd, scipy.integrate as scint
import time, itertools,sys
import seaborn as sns
from competition import generate_prm,Path,mpfig,make_point_measure,make_measures,code_debugger



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


path=Path('graddisp2')
plots=[
        #'all', #PLOT ALL TRAJECTORIES IN A SINGLE MULTIPLOT (REMOVE IF MANY VALUES OF d AND sigma)
        'points',
        'heatmap'
       ]
       
#### RELOAD ALREADY COMPUTED RESULTS
RERUN = 'rerun' in sys.argv
path.mkdir()
results = None
if not RERUN:
    try:
        results = pd.read_json(path+'results.json')
    except:
        pass

readonly=0
if 'read' in sys.argv and not results is None:
    versions=[{'d':d,'sigma':s} for d,s in results[['d','sigma']].values]
    if not ('remeasure' in sys.argv or 'rerun' in sys.argv):
        readonly=1
    for p in parameters:
        if p in results:
            if results[p].std()>0:
                code_debugger()
            print "SETTING",p, "FROM",parameters[p],"TO",results[p].mean()
            parameters[p]=results[p].mean().astype(type(parameters[p]))
else:
    versions = [{'d': d, 'sigma': sigma} for d, sigma in itertools.product(ds, sigmas)]            
            
#### GENERATING MATRICES
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

dists = np.add.outer(pos, -pos)
N0 = np.ones(K.shape) * 2
r,tmax,Nmin =[parameters[i] for i in ('r','tmax','Nmin')]


    
table=[] #WILL CONTAIN STORED RESULTS
for vidx, version in enumerate(versions):
    if readonly:
        break
    tref = time.time()
    label='_'.join(['{}:{:.2g}'.format(i,version[i]) for i in sorted(version)])
    print label
    row=None
    if not results is None and label in results['label'].values:
        row=results.set_index('label').loc[label ]
        if len(row.shape)>1:
            row=row.iloc[0]
        try:
            assert len(row['Nf'][0])==M
        except:
            code_debugger()
        row=row.to_dict()
        row['label']=label
        result=row['Nf']=np.array(list(row['Nf']))
        if not 'remeasure' in sys.argv:
            table.append(row )
            continue
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

    if row is None:
        integrator = scint.ode(eqs).set_integrator('dop853', nsteps=100000)

        integrator.set_initial_value(N0.ravel(), 0)
        result = integrator.integrate(tmax).reshape(K.shape)
  
    #MAKE MEASUREMENTS (STORE RESULTS)
    dic={'label':label,'Nf':result,'runtime': time.time() - tref}
    dic.update(parameters)
    dic.update(version)

    #MEASUREMENTS AT EACH POINT OF THE GRADIENT
    df=[]
    try:
        dNdt=eqs(tmax,result).reshape(K.shape) #derivatives
    except:
        code_debugger()
    for i, p in enumerate(pos):
        measure,locdata={},{'selfint':np.ones(S),'community':A,'growth':K[:,i]}
        measure.update({'position':p, 'Nf':result[:,i],'dNdt':dNdt[:,i] })
        make_point_measure(locdata, measure,death=1)
        df.append(measure)
    #COMBINED MEASUREMENTS OVER THE WHOLE GRADIENT
    make_measures(pd.DataFrame(df),dic,death=1)
    table.append(dic)

if not readonly:
    results = pd.DataFrame(table)
    results.to_json(path+'results.json')

results.reset_index()
#### PLOTS
if 'all' in plots or 'points' in plots:
    locres=results
    if 'points' in plots:
        sigs,ds=sorted(set(results['sigma'])),sorted(set(results['d']))
        points = [(ds[x],sigs[x]) for x in [0,10,20]]
        locres=results[[(d,s) in points  for d,s in results[['d','sigma']].values]]
    npanels=len(locres)
    side=np.ceil(np.sqrt(npanels)).astype('int')
    plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize'])/2.*(side,side))
    locres=locres.reset_index()
    for vidx, row in locres.iterrows():
        result = np.array(row['Nf'])
        plt.subplot(side, side, vidx + 1)
        x=np.linspace(100,200,result.shape[1])
        for y in result:
            plt.plot(x,y)
            plt.fill_between(x, y, 0,alpha=.1)
        plt.title(','.join(['{}={}'.format(i, row[i]) for i in ['d', 'sigma']]) + '\nruntime={:.2f}'.format(
            row['runtime']))
    pass
if 'heatmap' in plots:
    df=results
    df['stdJ'] = [np.std(x) for x in [np.array(z) for z in df['Jaccard'].values]]
    df['meandNs']=[np.max(x)<2 for x in df['dNs'].values]
    def makegini(dNs):
        dNs=np.array(dNs)
        return np.sum(np.abs(np.add.outer(dNs,-dNs)))/2/len(dNs)/np.sum(dNs)
    df['Gini']=[makegini(x) for x in df['dNs'].values]
    df['Giniabs']=df['Gini']*df['alive']
    locdf=df#[(df['d']<=3)&(df['sigma']<=11)]
    for val in ['stdJ','alive','Gini','Giniabs','meandNs']:
        showdf=locdf[['d','sigma',val]]
        tab=showdf.pivot(columns='d', index='sigma', values=val)
        vmax = np.max(tab.values)
        vmin= np.min(tab.values)
        plt.figure()
        if vmin<0 and vmax>0:
            vmax=np.max(np.abs([vmin,vmax]))
            vmin=-vmax
            cmap='seismic_r'
        else:
            cmap='OrRd'
        #sns.set()
        from bisect import bisect_right
        tshape=tab.shape[::-1]
        sns.heatmap(tab, vmax=vmax, vmin=vmin, cmap='OrRd',)
        xrg=np.sort(np.unique(showdf['d'].values))
        yrg=np.sort(np.unique(showdf['sigma'].values))
        xticks=np.linspace(-3,2,6)
        yticks=[1,5,10,15]
        xticklabels=['{:.1f}'.format(x) for x in xticks ] 
        yticklabels=['{:.1f}'.format(y)  for y in yticks] 
        print xticks, yticks, np.log10(xrg)
        xticks=[bisect_right(np.log10(xrg),x) for x in xticks]
        yticks=[bisect_right(yrg,y) for y in yticks]
        shape=tshape
        #plt.gca().invert_yaxis()
        plt.gca().set(xticks= xticks,yticks= yticks,xticklabels=xticklabels,yticklabels=yticklabels,
                      xlabel=r'$\mu(A)$',ylabel=r'$\sigma(A)$')
        sns.set_style("ticks")
        sns.despine(offset=10, trim=True)
         
        plt.title(val),plt.xlabel(r'$\log_{10}$ d'),plt.ylabel(r'$\sigma$')
        plt.gca().invert_yaxis()
        for p in [
                (0.001, 1),
                (0.046, 5.66),
                (2.15, 10.33)]:
            plt.scatter(bisect_right(xrg,p[0]),bisect_right(yrg,p[1]),c='k')
        plt.savefig(path + '{}.pdf'.format(val))
        
            
plt.show()


