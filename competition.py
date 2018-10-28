import os, sys, inspect,matplotlib.pyplot as plt, numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.figure as mpfig
from copy import deepcopy


## UTILITIES

def offdiag(x):
    return x[np.eye(x.shape[0])==0]

def rotmat(S):
    Nmi=np.random.random(S)
    Nmi/=la.norm(Nmi)
    rn = np.random.normal(0, 1, (S - 1, S - 1))
    tmp = np.concatenate([np.atleast_2d(Nmi[1:]), rn])
    AA = np.concatenate([np.atleast_2d(Nmi), tmp.T])
    QQ = la.qr(AA)[0].T
    QQ = QQ / np.sign(QQ[0, 0])
    return QQ

def setfigpath(path):
    import os
    import matplotlib as mpl
    mpl.rcParams["savefig.directory"] = os.path.dirname(path)

def code_debugger(skip=0):
    import code
    import inspect
    stack=inspect.stack()
    def trace():
        print('\n'.join([' '.join([str(x[y]) for y in [1, 2]]) + '\n      ' + str(not x[4] or x[4][0]).strip() for x in
                         stack if x[4]]))
    dic = {}
    dic.update(stack[1+skip][0].f_globals)
    dic.update(stack[1+skip][0].f_locals)
    dic['trace']=trace
    code.interact(local=dic)

class Path(str):
    '''Strings that represent filesystem paths.
    Overloads __add__:
     - when paths are added, gives a path
     - when a string is added, gives a string'''
    def __add__(self,x):
        import os
        if isinstance(x,Path):
            return Path(os.path.normpath(os.path.join(str(self),x)))
        return os.path.normpath(os.path.join(str(self),x))

    def norm(self):
        import os
        return Path(os.path.normpath(str(self)))

    def osnorm(self):
        """Deal with different separators between OSes."""
        import os
        if os.sep=='/' and "\\" in str(self):
            return Path(os.path.normpath(str(self).replace('\\','/' )))
        elif os.sep=='\\' and "/" in str(self):
            return Path(os.path.normpath(str(self).replace('/','\\' )))
        else:
            return self.norm()

    def prev(self):
        import os
        lst=self.split()
        path=os.path.join(lst[:-1])
        return path.osnorm()

    def split(self):
        """"""
        import os
        lst=[]
        cur=os.path.split(self.norm())
        while cur[-1]!='':
            lst.insert(0,cur[-1])
            cur=os.path.split(cur[0])
        return lst

    def mkdir(self,rmdir=False):
        """Make directories in path that don't exist. If rmdir, first clean up."""
        import os
        if rmdir:
            os.rmdir(str(self))
        cur=Path('./')
        for intdir in self.split():
            cur+=Path(intdir)
            if not os.path.isdir(cur):
                os.mkdir(cur)

    def copy(self):
        return Path(self)

    def strip(self):
        '''Return string without final / or \\ to suffix/modify it.'''
        return str(self).strip('\/')

## CODE

def generate_prm(S=50,Krange=(80,120),Trange=(20,30),symm=1,gfull=(0,300),arange=(0,1),mean=.5,wid=.5,**kwargs):
    # 50 sp
    # Distribution uniforme des coefficients, interactions symm, diag = 1, pas d'interaction negative ni >1
    # Distribution uniforme des Kmax entre 80 et 120
    # Distribution uniforme de la tolerance entre 15 et 30
    assert mean>0
    wid=min([wid,arange[1]-mean,mean-arange[0]])
    vals=np.sort(np.random.uniform( mean-wid,mean+wid ,S*(S-1)/2 ))
    mn=len(vals)/2.
    ref=np.arange(0,len(vals) ).astype('float')
    partners=np.random.permutation(ref)
    if symm<0:
        ref=ref[::-1]
        symm=np.abs(symm)
    dst=(ref-partners)
    partners+=symm*dst
    vals2=vals[np.argsort(np.argsort(partners))]
    idxs=np.random.permutation(range(len(vals)))
    alpha=np.ones((S,S))
    tru=(np.triu(alpha,k=1)>0)
    alpha[tru]= vals[idxs]
    alpha=alpha.T
    alpha[tru]= vals2[idxs]
    alpha = alpha.T
    Kmax=np.random.uniform(*Krange,size=S)
    lopt=np.random.uniform(*gfull,size=S)
    tol=np.random.uniform(*Trange,size=S)
    data={'alpha':alpha,'Kmax':Kmax, 'lopt':lopt,'tol':tol }
    mode=kwargs.pop('mode',None)
    if mode == 'alpha':
        data['alpha2'] = a = generate_prm(S=S, mean=.5, wid=.5, symm=symm, gfull=gfull, **kwargs)['alpha']
        data['phase'] = np.random.random(a.shape)
        data['freq'] = np.random.uniform(0.1,1,size= a.shape)
    return data

def make_point_measure(data,measure,**kwargs):
    """Measures at one point of the gradient"""
    N=measure['Nf']
    Aij=data['community']
    D=data['selfint']
    g=data['growth']
    death=1#kwargs.get('death',10**-6)
    alive=np.where(N>death)[0]
    Slive=len(alive)
    Alive=Aij[np.ix_(alive,alive)]
    glive,Dlive=g[alive],D[alive]
    Nlive=N[alive]
    B=(Alive-np.diag(Dlive) )
    diag=np.diag(Nlive)
    J=np.dot(diag,B)
    try:
        Press=-la.inv(B)#*Nlive# - np.eye(Slive)
    except:
        Press=np.zeros(B.shape)

    measure['W']= g+ np.dot(Aij-np.diag(D),N)
    measure['eigdom_J']=np.max(np.real(la.eigvals(J)))/np.mean(Nlive)
    measure['eigdom']=np.max(np.real(la.eigvals(B)))
    if measure.get( 'eigdom_full',None) is None:
        measure['eigdom_full']=np.max(np.real(la.eigvals(Aij-np.diag(D))))
    measure['alive']=alive
    measure['Press']=Press
    measure['eqdist']=eqdist=la.norm(np.dot(B,Nlive)+glive)/Slive

def make_run(data,tmax=2000,tsample=100,death=10**-6,mode='K',length=10,grange=(100,200),init=2,reverse=0,**kwargs):
    import scipy.integrate as scint
    interactions,Kmax,lopt, tol = [data[z] for z in ('alpha','Kmax','lopt','tol')]
    gradient = np.linspace(grange[0],grange[1], length + 1)
    S=Kmax.shape[0]
    selfint=np.ones(S)
    table = []
    result=None
    if reverse:
        gradient=gradient[::-1]
    measure={}
    transfer_measures=[]
    if mode!='alpha':
        """If interactions don't change, do not repeat measure of eigenvalues of full matrix"""
        transfer_measures.append('eigdom_full')
    for l in gradient:
        print l
        measure = {x:measure.get(x,None) for x in transfer_measures }
        capa = np.ones(tol.shape) * Kmax
        alpha=interactions
        if mode == 'K':
            capa = Kmax * np.exp(-(l - lopt) ** 2 / (2 * tol) ** 2)
            measure['K'] = capa
        elif mode == 'init':
            x0 = Kmax * np.exp(-(l - lopt) ** 2 / (2 * tol) ** 2)
            measure['x0'] = x0
        elif mode=='alpha':
            phase,freq=data['phase'],data['freq']
            lrel=(l*1.-np.min(gradient))/(np.max(gradient)-np.min(gradient))
            alpha=interactions+(data['alpha2']-interactions)*np.cos(np.pi*(phase+freq*lrel))**2
            measure['alpha'] = alpha

        if hasattr(init,'__iter__'):
            x0=init
        elif init == 'uniform':
            x0 = np.ones(tol.shape) * 2  # + (np.random.random(tol.shape)-.5)*3.
        elif 'follow' in init and not result is None:
            x0 = result+ np.random.uniform(0,1,S)
        else:
            x0 = np.random.uniform(0,1,S) * np.max(Kmax)/(1+capa) + 2

        capa = np.clip(capa, 10 ** -9, None)
        locdata={}
        locdata.update(data)
        locdata['growth'] = capa
        locdata['selfint'] = selfint
        locdata['community'] = -alpha

        def derivative(t, x):
            res = x * (capa - np.dot(alpha, x) - x * selfint)
            res[x < 10 ** -10] = np.clip(res[x < 10 ** -10], 0, None)
            return res
        integrator = scint.ode(derivative).set_integrator('lsoda', rtol=10. ** -13,nsteps=5000000)  # ,min_step=10**-50)
        integrator.t = 0
        integrator.set_initial_value(x0, t=0)
        result = integrator.integrate(tmax)
        result[result < death] = death
        error = np.sum((derivative(0, result) / result)[result > 10 ** -10])
        # print error
        if not integrator.successful() or np.isnan(result).any() or error > 0.001:
            print 'WARNING: INTEGRATION FAILED'
            # result[:] = 0
            # code_debugger()
            result=x0
        result[result <= death] = 0
        measure['position'] = l
        measure['Nf']=result
        make_point_measure(locdata,measure,death=death,**kwargs)

        table.append(measure)
    df=pd.DataFrame(table)
    return df


def make_measures(df,measure,**kwargs):
    '''Measure over a whole gradient (contained in df)'''
    df = df.sort_values('position')
    live=df['alive'].values
    Ns = [np.array(n) for n in df['Nf']]
    S=Ns[0].shape[0]
    Nlive=[np.array(n[a]) for n,a in zip(Ns,live) ]
    Nlive=[n for a in Nlive for n in a]
    # live=[np.where(n>0)[0] for n in Ns]
    measure.update({'N':np.mean(Ns),'Nstd':np.std(Ns), 'alive':np.mean([1.*len(a)/S for a in df['alive']]) ,'S':S,
                    'Nlive':np.mean(Nlive), 'Nlivestd':np.std(Nlive) })
    jacs = [1 - len(set(a1).intersection(a2)) * 1. / len(set(a1).union(a2)) for a1, a2 in zip(live[:-1], live[1:])]
    jacs=np.array(jacs)
    dNs = [la.norm( (n1-n2)/(10**-15+n1+n2)) for n1, n2 in zip(Ns[:-1], Ns[1:])]
    if 0:
        changes=[ pos for pos, a1, a2 in zip(df['position'].values[:-1], live[:-1], live[1:]) for i in set(a1).symmetric_difference(a2)]
        xch=(np.array(changes,dtype='float')-df['position'].min())/(df['position'].max() - df['position'].min())
        xch=np.sort(xch)
        K= np.mean(np.abs(xch - np.linspace(0,1,xch.shape[0])) )
        Krand=[np.mean(np.abs(np.sort(np.random.uniform(0,1,xch.shape[0])  )- np.linspace(0,1,xch.shape[0])) ) for test in range(1000) ]
        K/=np.mean(Krand)
        measure['uniform']=K
    measure['dNs']=dNs
    measure['uniform']=np.std(dNs)/np.mean(dNs+10**-15)
    rank=np.argsort(np.argsort(dNs))
    measure['Gini']=np.sum( ( 2*rank - len(dNs)-1)*dNs )/len(dNs)/np.sum(dNs)

    Vpos=np.mean([np.mean(offdiag(np.array(x))>0) for x in df['Press'].values  ] )
    Vstd=np.mean([np.std(offdiag(np.array(x))) for x in df['Press'].values  ] )
    Vdiagmin=np.min([np.min(np.diag(np.array(x))) for x in df['Press'].values  ] )
    Vdiagmax=np.max([np.max(np.diag(np.array(x))) for x in df['Press'].values  ] )
    Vdiagstd=np.mean([np.std(np.diag(np.array(x))) for x in df['Press'].values  ] )
    if np.isnan(Vpos):
        Vpos,Vstd=0,0
    # measure['Vcascade']= np.mean([np.max( np.max(x*y[a],axis=1)/y[a]) for row in df[['Press','Nf','alive']].values for x,y,a in ([np.array(z) for z in row],) ] )
    measure['Vcascade']= np.mean([np.max( np.max( (x-np.diag(np.diag(x))) *y[a])/np.max(y[a])) for row in df[['Press','Nf','alive']].values for x,y,a in ([np.array(z) for z in row],) ] )
    # code_debugger()

    for v in ('eigdom','eigdom_full','eigdom_J'):
        measure[v]=df[v].mean(skipna=1)
        measure[v+'_max']=df[v].max(skipna=1)

    measure.update( {'eqdist':df['eqdist'].max(), 'Jaccard':jacs,'Vpositive':Vpos,'Vstd': Vstd,'Vdiagmax':Vdiagmax,'Vdiagmin':Vdiagmin,'Vdiagstd':Vdiagstd })
    if 'reverse_Nf' in df:
        x1,x2=df['Nf'],df['reverse_Nf']
        x1,x2=np.array(list(x1.values)),np.array(list(x2.values))
        measure['hysteresis']=np.sum( 2*np.abs(x1-x2 )/ (x1+x2+1)  )
    else:
        measure['hysteresis']=0
    return measure


def loop(path='gradcomp',rerun=1,remeasure=0,resolution=11,S=30,gfull=(0,300),triangle='',
         systems=(0,),keep_sys=1, #Replicas
         symm=1,mode='K',**kwargs):
    path=Path(path)
    datarefs={}
    def make_data():
        dataref = generate_prm(S=S, mean=.5, wid=.5, symm=symm,gfull=gfull,mode=mode,**kwargs)
        return dataref
    for sys in systems:
        datarefs[sys]=make_data()
    vals=np.linspace(0,1,resolution)

    df =None# pd.DataFrame({'mean':[],'wid':[] })
    if not rerun and not remeasure:
        try:
            df = pd.read_json(path + 'measures.csv')
        except:
            pass
    for sys in systems:
        print '   Sys', sys
        for x in range(resolution):#[::-1]:
            for y in range( x+1):
                mn, wid = vals[x], vals[y]
                if wid > 1 - mn + 10 ** -6 and not triangle == 'rectangle':
                    continue
                if not rerun and not remeasure and not df is None and True in [np.allclose([mn, wid,sys], z) for z in
                                                                               df[['mean', 'wid','sys']].values]:
                    continue
                print 'MEAN {} WID {}'.format(mn, wid)
                if keep_sys:
                    data=deepcopy(datarefs[sys])
                else:
                    data=make_data()
                for i in data:
                    if 'alpha' in i:
                        mat=data[i]
                        mat=mn+wid*(mat-.5)
                        np.fill_diagonal(mat,0)
                        data[i]=mat
                dpath=path+Path('mn_{}-wd_{}-sys_{}'.format(mn,wid,sys) )
                dpath.mkdir()
                tdf=None
                fname=dpath+'traj.csv'
                measure={'path':dpath, 'mean':mn,'wid':wid,'sys':sys}
                if not rerun:
                    try:
                        tdf=pd.read_json(fname)
                    except Exception as e:
                        pass
                if tdf is None:
                    if remeasure:
                        continue
                    if kwargs.get('init',None)=='hysteresis':
                        kw={}
                        kw.update(kwargs)
                        kw['init']='follow'
                        tdf=make_run(data,mode=mode,ext_measure=measure,**kw)
                        # kw['init']=tdf['Nf'].values[-1]+ np.random.uniform(0,1,S)
                        tdf2=make_run(data,mode=mode,ext_measure=measure,reverse=1,**kw)
                        tdf['reverse_Nf']=tdf2['Nf'].values[np.argsort(tdf2['position'].values )]
                    else:
                        tdf=make_run(data,mode=mode,ext_measure=measure,**kwargs)
                    tdf.to_json(fname)

                # measure.update(m.export_params())
                make_measures(tdf,measure)
                if not df is None:
                    df=pd.concat([df,pd.DataFrame([ measure ]) ],ignore_index=1)
                else:
                    df=pd.DataFrame([ measure ])
    df.to_json(path+'measures.csv')




def local_plots(df,measure,fpath='',**kwargs):
    path=Path(fpath)
    profiles=[np.array(list(df['Nf'].values)).T]
    if 'reverse_Nf' in df:
        profiles.append(np.array(list(df['reverse_Nf'].values)).T)
        hyster=1
    else:
        hyster=0
    ii=0
    plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize'])*(len(profiles)+hyster,1))
    S = len(profiles[0])
    vec=np.random.random(S)
    for profile in profiles:
        ii+=1
        if len(profiles)>1:
            plt.subplot(1,len(profiles)+hyster,ii)
        pos = df['position']
        for y in profile:
            x,y=np.sort(pos),y[np.argsort(pos)]
            plt.plot(x,y)
            plt.fill_between(x, y, 0,alpha=.1)
        if hyster:
            plt.subplot(1,len(profiles)+1,len(profiles)+1)
            y=np.dot(vec,profile)
            x, y = np.sort(pos), y[np.argsort(pos)]
            plt.plot(x,y)
    if kwargs.get('save',1):
        plt.savefig(path + 'profile.png')
        plt.close()


def detailed_plots( path,**kwargs):
    print 'Detailed plot',Path(path)+'measures.csv'
    df=pd.read_json(Path(path)+'measures.csv')
    if 'filter' in kwargs:
        df = df.query(kwargs['filter'])
    df=df.sort_values('path')
    figs=[]
    for idx,measure in df.iterrows():
        dpath=measure['path']
        print '  ...Plotting',dpath
        tdf=pd.read_json(Path(dpath)+'traj.csv')
        figs.append(local_plots(tdf, measure, fpath=dpath,**kwargs))
    return figs

def show(path='gradcomp',detailed=0,hold=0,**kwargs):
    """Summary plots"""

    import seaborn as sns
    sns.set(style="white")
    path=Path(path)
    df=pd.read_json(path + 'measures.csv')
    df['J=1']= [np.mean(x==1) for x in [np.array(z) for z in  df['Jaccard'].values]]
    df['meanJ']= [np.mean(x[x>0]) for x in [np.array(z) for z in  df['Jaccard'].values]]
    df['stdJ']= [np.std(x) for x in [np.array(z) for z in  df['Jaccard'].values]]
    # df['relstdJ']= df['stdJ']/[np.mean(x) for x in [np.array(z) for z in  df['Jaccard'].values]] #NOT GOOD
    df['hysteresis']=np.clip(np.log10(10**-10+df['hysteresis']),-1,None)
    df['hysteresis>1']=[float(x>1.) for x in df['hysteresis']]
    df['eigdom_full>0']=[float(x>0) for x in df['eigdom_full_max']]
    df['eigdom_max']=np.clip(df['eigdom_max'],-1,None)
    df['eigdom_J_max']=np.clip(df['eigdom_J_max'],-.05,None)
    df['alone']=[1.*(a*len(n)<2) for a,n in df[['alive','dNs']].values  ]
    for N in ('N','Nlive'):
        df[N+'relstd']=df[N+'std']/df[N]
    axes=['mean','wid']
    mode=kwargs.get('mode','dft')
    if mode =='full':
        vals=['Vpositive','Vcascade','Vstd','Vdiagmin',#'Vdiagmax','Vdiagstd',
              'meanJ','stdJ',#'relstdJ',#'J=1',
              'uniform','Gini',
              'eigdom_max','eigdom', 'eigdom_J_max','eigdom_full_max',  'eigdom_full>0',
              # 'eqdist',
              'alone','Nliverelstd',#'Nrelstd',
              'hysteresis']
    else:
        vals=['alone','hysteresis','Vstd','stdJ','eigdom_full>0']
    vals=[v for v in vals if v in df.keys()]
    showdf=df[axes+vals].groupby(axes).mean().reset_index()
    for val in vals:
        plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize'])*(1,.5)),plt.title(val)
        if 'eigdom_max' in val:
            sns.heatmap(showdf.pivot(columns='mean', index='wid', values=val),vmax=0, cmap='PuRd')
        else:
            sns.heatmap(showdf.pivot(columns='mean', index='wid', values=val),cmap='PuRd')
        plt.gca().invert_yaxis()

    plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize']) * (1.5, .5)), plt.title(val)
    points=[(.1,.1), (.5,.25),(0.5,0.5),(.9,.1) ]
    points=[(0,0), (.5,.25),(0.5,0.5),(1,0) ]
    for ip,p in enumerate(points):
        plt.subplot(1,len(points),ip+1)
        mw=df[['mean','wid']].values
        closest=mw[np.argmin([la.norm(v-p) for v in mw] )]
        pdf=df[(df['mean']==closest[0]) & (df['wid']==closest[1])]
        Js=np.concatenate(pdf['Jaccard'].values)
        plt.hist(Js,bins=np.linspace(0,1,10)),plt.title('Mean {} wid {}'.format(*closest) )

    if not hold:
        plt.show()
    for ip, p in enumerate(points):
        mw=df[['mean','wid']].values
        closest=mw[np.argmin([la.norm(v-p) for v in mw] )]
        detailed_plots(path, filter='mean=={} & wid=={} & sys==1'.format(closest[0],closest[1]),save=0 )
    if not hold:
        plt.show()
        code_debugger()



if __name__=='__main__':
    #NOTES:

    # DEFAULT OPTIONS
    default={'resolution':21, # Resolution along x-axis in sampling triangle of interaction mean and sd
             'S':50, #Number of species
             'length':50,    # Number of positions along gradient
             'systems':(0,), #Replicas (list of labels e.g. ['a','b','c'] or range(3) for 3 replicas)
             'init': 'hysteresis', # Initial conditions
                    # ('uniform' for the same everywhere, 'random' for random,
                    #  'follow' to follow an eq, 'hysteresis' to follow forward then backward)
             'keep_sys':1   # Keep same basic matrix throughout triangle for each replica (seems to give smoother visuals)
             }

    # SETS OF OPTIONS FOR DIFFERENT SIMULATION RUNS
    runs={'default':{},
          'alpha':{'mode':'alpha'}, #Change interactions rather than carrying capacities
          'rectangle':{'triangle':'rectangle'}, #Explore half rectangle
          'nokeep':{'keep_sys':0, 'systems':range(5)}  # Generate new properties at every point in the triangle
          }
    for symm in [-1,0,1]:
        runs['sym{}'.format(symm) ]={'symm':symm,'resolution':41,'systems':range(5)}

    run =None
    for i in sys.argv:
        if i in runs:
            run=i
    if run is None:
        run='sym1' #Change this to switch between different runs (put 'default' for default optons)

    def do(run,with_plots=1):
        options={}
        options.update(default)
        if '+' in run:
            for r in run.split('+'):
                options.update(runs[r] )
        else:
            options.update(runs[run])
        path='gradcomp_'+run
        if not 'show' in sys.argv:
            loop(path=path, rerun='rerun' in sys.argv, remeasure='measure' in sys.argv,**options)
        if not 'show' in sys.argv or 'detailed' in sys.argv:
            detailed_plots(path, save=1)#,filter='mean==.5 & wid==.5')
        if with_plots:
            setfigpath(Path(path) )  # Path for saving figures
            show(path,mode='full')

    if 'multirun' in sys.argv:
        for run in ['alpha','rectangle','sym0','sym-1']:
            do(run,with_plots=0)
    else:
        do(run)