import os, sys,time, inspect,matplotlib.pyplot as plt, numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.figure as mpfig
from copy import deepcopy


## UTILITIES
from numpy import sqrt,pi,exp,log,sin,cos
from scipy.special import erf,binom

#==================== BASICS ============================

def erfmom(mom,mean,var,lim=0):
    #Moment of error function
    if var<=0.001:
        var=0.001
    xx=mean/sqrt(2*var)
    mom0=.5 * (erf(xx)+1 )
    mom1=sqrt(var/2/pi) * exp(-xx**2)
    if mom==0:
        return mom0
    elif mom==1:
        return mean* mom0 + mom1
    elif mom==2:
        return (var+mean**2)*mom0 + mean*mom1

def bunin_solve(S=100,mu=1,sigma=1,sigma_k=1,gamma=1,tol=10**-5,**kwargs):
    import scipy.optimize as sopt
    u=(1-mu*1./S)
    def calc_v(phi):
        psg=np.clip(phi*sigma**2*gamma,None,u/2)
        if psg<10**-6:
            v=1./u
        else:
            v= (u-np.sqrt(u**2-4 *psg ))/(2*psg)
        return v
    def eqs(vec):
        N1,N2=np.abs(vec[:2])  #N1, N2 must be positive
        phi=np.exp(vec[2])/(1+np.exp(vec[2])) #phi must be between 0 and 1
        v=calc_v(phi)
        mean=1-v*mu*N1*phi
        var=np.clip(v**2*(sigma_k**2+sigma**2*N2*phi),0.001,None)
        m0,m1,m2=erfmom(0,mean,var),erfmom(1,mean,var),erfmom(2,mean,var)
        eq1=(phi-m0)
        eq2=(phi*N1-m1)
        eq3=(phi*N2-m2)
        res= np.array( (eq1,eq2,eq3))
        return res
    x0=kwargs.get('x0',np.random.random(3))
    res= root(eqs,x0,tol=tol)  #ROOTFINDER
    if not res.success:
        #IF ROOT FINDING DOES NOT WORK (MAYBE DIVERGENCE, MAYBE BAD INITIAL CONDITION)
        if sigma>.5 and mu>-1:
            #IF LARGE SIGMA, GET INITIAL CONDITION FROM SLIGHTLY LOWER SIGMA
            x0=bunin_solve(S=S,mu=mu,sigma=sigma/1.01,sigma_k=sigma_k,gamma=gamma,tol=tol)[:3]
            res=root(eqs,x0,tol=tol)
        else:
            #ELSE, JUST TRY A FEW TIMES THEN QUIT
            trials=50
            while not res.success and trials>0:
                x0=np.random.random(3)
                res= root(eqs,x0,tol=tol)
                trials-=1
    N1, N2, phi = res.x
    N1,N2=np.abs(N1),np.abs(N2)
    phi=np.exp(phi)/(1+np.exp(phi))
    if not res.success or N1>5:
        return np.nan,np.nan,np.nan,np.nan
    return N1,N2,phi, calc_v(phi)


def offdiag(x):
    return x[np.eye(x.shape[0])==0]

def ifelse(x,a,b):
    if x:
        return a
    return b

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

def generate_prm(S=50,Krange=(80,120),Trange=(15,30),symm=1,gfull=(0,300),arange=(0,1),mean=.5,wid=.5,**kwargs):
    # 50 sp
    # Distribution uniforme des coefficients, interactions symm, diag = 1, pas d'interaction negative ni >1
    # Distribution uniforme des Kmax entre 80 et 120
    # Distribution uniforme de la tolerance entre 15 et 30
    assert mean>0
    wid=min([wid,arange[1]-mean,mean-arange[0]])
    if kwargs.get('distribution','uniform'):
        vals=np.sort(np.random.uniform( mean-wid,mean+wid ,S*(S-1)/2 ))
    else:
        sd=wid/np.sqrt(3)
        vals = np.sort(np.random.normal(mean,sd, S * (S - 1) / 2))
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
    # vals = np.sort(np.random.uniform(mean - wid, mean + wid, (S,S)))
    # alpha=(vals+vals.T)/2
    np.fill_diagonal(alpha,0)
    # plt.hist(offdiag(alpha)),plt.show()
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
    S=len(N)
    Aij=data['community']
    D=data['selfint']
    g=data['growth']
    K=g/D
    measure['capa']=K
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
    eigJ=np.max(np.real(la.eigvals(J)))
    measure['eigdom_J']=eigJ
    measure['eigdom_Jscaled']=eigJ/np.mean(Nlive)
    measure['eigdom']=np.max(np.real(la.eigvals(B)))
    Bfull=Aij-np.diag(D)
    if measure.get( 'eigdom_full',None) is None:
        measure['eigdom_full']=np.max(np.real(la.eigvals(Bfull)))
    if measure.get( 'eigdom_fullT',None) is None:
        measure['eigdom_fullT']=np.max(np.real(la.eigvals(Bfull+Bfull.T)))
    if measure.get('feasible', None) is None:
        try:
            measure['feasible']=np.mean(np.dot(-la.inv(Aij-np.diag(D)),data['Kmax'] )>0)
        except:
            measure['feasible']=0

    measure['alive']=alive
    measure['%alive']=len(alive)*1./S
    measure['Press']=Press
    measure['eqdist']=np.max(np.abs(np.dot(B,Nlive)+glive))
    meanA, stdA = np.mean(offdiag(Aij)), np.std(offdiag(Aij))
    measure['mu'],measure['sigma']= -S*meanA,np.sqrt(S)*stdA
    measure['meanK'],measure['stdK']=np.mean(K),np.std(K)

    #MEAN FIELD PREDICTIONS
    def MF(K):
        S=len(K)
        avgNMF=np.mean(K)/(1+(S-1) *meanA)
        NMF=K - meanA*avgNMF*(S-1)
        if np.min(NMF)<0 and S>1:
            return MF(K[K>np.min(K) ])
        return avgNMF, NMF, S
    measure['MF_avgN'],measure['MF_N'],measure['MF_Slive']=MF(K)


def make_run(data,tmax=20000,tsample=100,death=10**-6,mode='K',length=10,grange=(100,200),init=2,reverse=0,**kwargs):
    import scipy.integrate as scint
    interactions,Kmax,lopt, tol = [data[z] for z in ('alpha','Kmax','lopt','tol')]

    # np.savetxt('A.txt',interactions),np.savetxt('Os.txt',Kmax),np.savetxt('Cs.txt',lopt),np.savetxt('Ts.txt',tol)
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
        transfer_measures+=['eigdom_full','eigdom_fullT','feasible']
    for l in gradient:
        print l
        measure = {x:measure.get(x,None) for x in transfer_measures }
        capa = np.ones(tol.shape) * Kmax
        alpha=interactions
        if mode == 'K':
            capa = Kmax * np.exp(-(l - lopt) ** 2 / (2 * tol** 2) )
            measure['K'] = capa
        elif mode == 'init':
            x0 = Kmax * np.exp(-(l - lopt) ** 2 / (2 * tol** 2) )
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
            x0 = np.random.uniform(0,1,S) * np.max(Kmax) + 2  #/(1+capa)

        capa = np.clip(capa, 10 ** -9, None)
        locdata={}
        locdata.update(data)
        locdata['growth'] = capa
        locdata['selfint'] = selfint
        locdata['community'] = -alpha
        locdata['Kmax']=Kmax

        def derivative(t, x):
            res = x * (capa - np.dot(alpha, x) - x * selfint)
            res[x < 10 ** -10] = np.clip(res[x < 10 ** -10], 0, None)
            return res
        integrator = scint.ode(derivative).set_integrator('lsoda', rtol=10. ** -13,nsteps=5000000)  # ,min_step=10**-50)
        integrator.t = 0
        integrator.set_initial_value(x0, t=0)
        result = integrator.integrate(tmax)
        result[result < death] = death
        error = np.sum((derivative(0, result) / result)[result > death])
        if not integrator.successful() or np.isnan(result).any() or np.abs(error) > 0.001:
            print 'WARNING: INTEGRATION FAILED',error
            # result[:] = 0
            code_debugger()
            result=x0
        result[result <= death] = 0
        measure['position'] = l
        measure['Nf']=result
        measure['dNdt']= derivative(0,result)
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
    measure.update({'N':np.mean(Ns),'Nstd':np.std(Ns), 'alive':np.mean([1.*len(a)/S for a in live]) ,'S':S,
                    'Nlive':np.mean(Nlive), 'Nlivestd':np.std(Nlive) })
    jacs = [1 - len(set(a1).intersection(a2)) * 1. / len(set(a1).union(a2)) for a1, a2 in zip(live[:-1], live[1:])]
    jacs = [len(set(a1).union(a2)) - len(set(a1).intersection(a2))  for a1, a2 in zip(live[:-1], live[1:])]
    jacs=np.array(jacs)*1./np.mean([len(a) for a in live])
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
    measure['uniform']=np.std(dNs)/np.mean(np.array(dNs)+10**-15)
    rank=np.argsort(np.argsort(dNs))
    measure['Gini']=np.sum( ( 2*rank - len(dNs)-1)*dNs )/len(dNs)/np.sum(dNs)
    measure['feasible']=np.mean(df['feasible'])
    measure['stable']=np.mean(df['eigdom_full']<0)
    measure['Gleason']=measure['feasible'] * measure['stable']

    Vpos=np.mean([np.mean(offdiag(np.array(x))>0) if len(x)>1 else 0 for x in df['Press'].values  ] )
    Vstd=np.mean([np.std(offdiag(np.array(x))) if len(x)>1 else 0 for x in df['Press'].values   ] )
    Vmax=np.mean([np.max(offdiag(np.array(x))) if len(x)>1 else 0 for x in df['Press'].values   ] )
    Vdiagmin=np.min([np.min(np.diag(np.array(x))) for x in df['Press'].values  ] )
    Vdiagmax=np.max([np.max(np.diag(np.array(x))) for x in df['Press'].values  ] )
    Vdiagstd=np.mean([np.std(np.diag(np.array(x))) for x in df['Press'].values  ] )
    # measure['Vcascade']= np.mean([np.max( np.max(x*y[a],axis=1)/y[a]) for row in df[['Press','Nf','alive']].values for x,y,a in ([np.array(z) for z in row],) ] )
    measure['Vcascade_worst']= np.mean([ np.max( (x-np.diag(np.diag(x))) *y[a])/np.max(y[a]) for row in df[['Press','Nf','alive']].values for x,y,a in ([np.array(z) for z in row],) ] )
    measure['Vcascade_max']= np.max([ np.max( (x-np.diag(np.diag(x))) *y[a]/y[a]) for row in df[['Press','Nf','alive']].values for x,y,a in ([np.array(z) for z in row],) ] )
    measure['Vcascade']= np.mean([ np.max( (x-np.diag(np.diag(x))) *y[a]/y[a]) for row in df[['Press','Nf','alive']].values for x,y,a in ([np.array(z) for z in row],) ] )
    # code_debugger()


    #BUNIN PREDICTION OF TRANSITION
    sigma=np.sqrt(S)*measure['wid']/np.sqrt(3)#/(1.001-measure['mean'])
    u=(1-measure['mean'])
    V=np.mean([np.mean(np.diag(np.array(x))) for x in df['Press'].values  ] )
    gamma=measure['sym']
    measure['v']=V
    if 'MF_Slive' in df and 0:
        meanK = df['meanK'].mean()
        if sigma>10**-5:
            sigma=df['sigma'].mean()
            N1,N2,phi,v= bunin_solve(S=S,mu=df['mu'].mean(),sigma=sigma,sigma_k=df['stdK'].mean()/meanK,gamma=gamma)
        else:
            v,phi=1./np.sqrt(u ),df['MF_Slive'].mean()/S
    else:
        phi=measure['alive']
        v=2/(u +np.sqrt(u**2-4*gamma*sigma**2*phi) )
    measure['v_from_phi']=v#
    def safelog(x):
        if x>0:
            return np.log10(x)
        return np.nan
    # measure['bunin']=safelog(chi )  #np.abs( sigma**2 - 1./phi*(1-1.*measure['mean'])**2/(1+gamma)**2 )- np.abs(sigma**2)
    measure['bunin'] = (1./v**2 - phi*sigma**2 )
    # code_debugger()
    phiobs=measure['alive']
    chi2=(u-phiobs*gamma*V*sigma**2)**2 - phiobs*sigma**2
    measure['bunincomp']=(1./V**2 - phiobs*sigma**2 )#safelog(chi2)
    measure['bunincomp2']=chi2

    for v in ('eigdom','eigdom_full','eigdom_fullT','eigdom_Jscaled'):
        if not v in df.keys():
            continue
        measure[v]=df[v].mean(skipna=1)
        measure[v+'_max']=df[v].max(skipna=1)

    measure.update( {'eqdist':df['eqdist'].max(), 'Jaccard':jacs,'Vpositive':Vpos,'Vstd': Vstd,'Vmax':Vmax,'Vdiagmax':Vdiagmax,'Vdiagmin':Vdiagmin,'Vdiagstd':Vdiagstd })
    if 'reverse_Nf' in df:
        x1,x2=df['Nf'],df['reverse_Nf']
        x1,x2=np.array(list(x1.values)),np.array(list(x2.values))
        measure['hysteresis']=np.sum( 2*np.abs(x1-x2 )/ (x1+x2+1)  )
    else:
        measure['hysteresis']=0

    try:
        measure['eqrate']=  np.max([np.max(np.abs(np.array(d)/n))/l for d,l,n in zip(df['dNdt'].values,df['eigdom_J'].values,Ns) ])
    except:
        pass

    try:
        measure['GleasonNK']=np.mean([np.sum((np.array(N)>1)*1.*(np.array(K)>1))/np.sum(np.array(K)>1) for N,K in df[['Nf','capa']].values ])
    except:
        pass

    #Feasibility cone (Grilli et al 2015)
    m,s,gamma=measure['mean'],measure['wid']/np.sqrt(3),measure['sym']
    denom=1+(S-2)*(m**2+s**2)
    avgcoseta=2*m + S*m**2
    stdcoseta=np.sqrt(np.clip(2*(1+gamma)*s**2 + S*(m**2 + s**2)**2 - S*m**4,0,None))
    avgcoseta,stdcoseta=avgcoseta/denom,stdcoseta/denom+10**-5
    from scipy.special import erf
    measure['negarc']=(erf( (avgcoseta-1)/np.sqrt(2)/stdcoseta )+1) /2 # np.exp(-(avgcoseta-denom)**2/(2*stdcoseta**2) )/np.sqrt(2*pi*stdcoseta**2/denom**2)
    from numpy import pi
    d=-1
    xie=S*(2*(pi-1)*m**2 + pi * s**2) - d*(2*(pi-2) *m +pi  )
    xi1=pi * m*(2*d-m*S)*(2*d*m+d - S*(2*m**2+s**2) )
    xi2=np.sqrt(pi) *m *(m*S-2*d)
    measure['xi']=xi1/xie**2 * np.log( 1-erf(xi2/xie) )  + np.log(1+ (3*S*s**2*(1+gamma))/(2*pi) )
    beta=S*(2*m + (S-2)*m**2) / (1 + (S-1)*m**2)
    measure['xi_simple']=np.log(np.clip(1-beta/S/pi,10**-15,None))
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
                al=data['alpha']
                gamma=np.corrcoef(offdiag(al),offdiag(al.T))[0,1]
                if np.isnan(gamma):
                    gamma=0
                measure={'path':dpath, 'mean':mn,'wid':wid,'sym':gamma,
                         'sys':sys,'rectangle':triangle=='rectangle'}
                if not rerun:
                    try:
                        tdf=pd.read_json(fname)
                    except Exception as e:
                        pass
                if tdf is None:
                    if remeasure:
                        continue
                    init=kwargs.get('init',None)
                    if init =='hysteresis':
                        kw={}
                        kw.update(kwargs)
                        kw['init']='follow'
                        tdf=make_run(data,mode=mode,ext_measure=measure,**kw)
                        # kw['init']=tdf['Nf'].values[-1]+ np.random.uniform(0,1,S)
                        tdf2=make_run(data,mode=mode,ext_measure=measure,reverse=1,**kw)
                        tdf['reverse_Nf']=tdf2['Nf'].values[np.argsort(tdf2['position'].values )]
                    elif  init =='tworandom':
                        kw={}
                        kw.update(kwargs)
                        kw['init']='random'
                        tdf=make_run(data,mode=mode,ext_measure=measure,**kw)
                        tdf2=make_run(data,mode=mode,ext_measure=measure,reverse=1,**kw)
                        tdf['reverse_Nf']=tdf2['Nf'].values[np.argsort(tdf2['position'].values )]
                    else:
                        tdf=make_run(data,mode=mode,ext_measure=measure,**kwargs)
                    tdf['mean']=mn
                    tdf['wid']=wid
                    tdf['sys']=sys
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
    plt.suptitle(kwargs.get('title','Mn {} Wd {}'.format(df['mean'].mean(),df['wid'].mean()) ))
    if kwargs.get('save',1):
        plt.savefig(path + kwargs.get('fname', 'profile') +kwargs.get('format','.png') )
        plt.close()


def detailed_plots( path,df=None,**kwargs):
    print 'Detailed plot',Path(path)+'measures.csv'
    if df is None:
        df=pd.read_json(Path(path)+'measures.csv')
    if 'filter' in kwargs:
        df = df.query(kwargs['filter'])
    df=df.sort_values('path')
    figs=[]
    for idx,measure in df.iterrows():
        dpath=measure['path']
        print '  ...Plotting',dpath
        tdf=pd.read_json(Path(dpath)+'traj.csv')
        kw={}
        kw.update(kwargs)
        fpath=kw.pop('fpath',dpath)
        fname = kw.pop('fname', 'profile')
        if 'fpath' in kwargs:
            fname=fname + '_{}'.format(idx)
        figs.append(local_plots(tdf, measure, fpath=fpath,fname=fname ,**kw))
    return figs

def show(path='gradcomp',detailed=0,hold=0,**kwargs):
    """Summary plots"""

    import seaborn as sns
    sns.set(style="white")
    path=Path(path)
    df=pd.read_json(path + 'measures.csv')
    if kwargs.get('triangle',0):
        df=df[df['wid']<=1-df['mean']]
    df['J=1']= [np.mean(x==1) for x in [np.array(z) for z in  df['Jaccard'].values]]
    df['meanJ']= [np.mean(x[x>0]) for x in [np.array(z) for z in  df['Jaccard'].values]]
    df['stdJ']= [np.std(x) for x in [np.array(z) for z in  df['Jaccard'].values]]
    # df['relstdJ']= df['stdJ']/[np.mean(x) for x in [np.array(z) for z in  df['Jaccard'].values]] #NOT GOOD
    df['hysteresis']= df['hysteresis']/(1.+df['hysteresis'])
    #df['hysteresis']=np.clip(np.log10(10**-10+df['hysteresis']),-1,None)
    df['hysteresis>1']=[float(x>1.) for x in df['hysteresis']]
    df['eigdom_full>0']=[float(x>0) for x in df['eigdom_full_max']]
    try:
        df['negdef']=[float(x>0) for x in df['eigdom_fullT_max']]
    except:
        print 'Old results, do not have negative definiteness'
    # df['stable']=[float(x<0) for x in df['eigdom_full_max']]
    df['eigdom_max']=np.clip(df['eigdom_max'],-1,None)
    df['eigdom_max~0']=np.clip([10*float(float(a)>-0.1) for a  in df['eigdom_max'].values ],0,None)
    df['alone']=[1.*(a*len(n)<2) for a,n in df[['alive','dNs']].values  ]
    for N in ('N','Nlive'):
        df[N+'relstd']=df[N+'std']/df[N]
    try:
        df['vcomp']=df['v']/df['v_from_phi']-1
        df['eigdom_Jscaled_max'] = np.clip(df['eigdom_Jscaled_max'], -.05, None)
    except:
        pass
    for key in df:
        if 'V' in key:# or 'bunin' in key:
            df[key]=np.clip(df[key],0,5)
    # df['bunincomp']=np.clip(df['bunincomp'],-1,1)
        # if 'bunin' in key:
            # df[key]=1./(1+df[key])
    # code_debugger()
    df['stdJlive']=df['stdJ']*df['alive']
    df['Vcascade']=np.clip(df['Vcascade'],0,1)
    df['Vcascade_max']=df['Vcascade_max']/(1+df['Vcascade_max'])
    df['Vstd']=np.clip(df['Vstd'],0,1)
    axes=['mean','wid']
    mode=kwargs.get('mode','dft')
    if mode =='full':
        vals=['Vpositive','Vcascade','Vstd','Vdiagmin',#'Vdiagmax','Vdiagstd',
              'meanJ','stdJ',#'relstdJ',#'J=1',
              'uniform','Gini',
              'eigdom_max','eigdom', 'eigdom_J_max','eigdom_full_max',  'eigdom_full>0',
              'negdef', 'negarc',
               'eqdist',
              'xi','xi_simple',
              'alone','Nliverelstd',#'Nrelstd',
              'bunin','bunincomp','bunincomp2',
              'hysteresis']
    else:
        vals=['bunincomp','Vpositive','hysteresis','stdJ','Gleason','negarc','eqdist']#'Vstd','Vcascade', 'stdJ','negdef','stable','Vpositive','eigdom_max']

    dico={'Vpositive':'Clements','bunincomp':'Phase parameter','hysteresis':'Multistability','negarc':'Gause','stdJ':'std(Jaccard)' }
    vals=[v for v in vals if v in df.keys()]
    showdf=df[axes+vals].groupby(axes).mean().reset_index()
    rectangle=showdf['wid'].max()>.5
    for val in vals:
        plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize'])*(1,(1+rectangle)*.5)),plt.title(val)
        if (np.min(showdf[val])<0 and np.max(showdf[val])>0) or 'eigdom' in val:
            vmax=np.max(np.abs(showdf[val]))
            sns.heatmap(showdf.pivot(columns='mean', index='wid', values=val),vmax=vmax,vmin=-vmax, cmap='seismic_r')
        else:
            sns.heatmap(showdf.pivot(columns='mean', index='wid', values=val),cmap='PuRd')
        plt.gca().invert_yaxis()
        plt.title(dico.get(val,val))
        plt.savefig(path+'{}.pdf'.format(val) )
    plt.figure(), plt.title('Criteria')
    crits=[showdf.pivot(columns='mean', index='wid', values=val).values.T for val in ['Gleason','Vpositive','negarc']]
    crit=np.array([c/np.max(c[~np.isnan(c)] ) for c in crits]).T
    crit[np.isnan(crit)]=1
    plt.imshow(crit)
    plt.gca().invert_yaxis()

    plt.figure(figsize=np.array(mpfig.rcParams['figure.figsize']) * (1.5, .5)), plt.title(val)
    points=[(.1,.1), (.5,.25),(0.5,0.5),(.9,.1) ]
    points=[(0.04,0.04), (.5,.25),(0.5,0.5),(.96,0.04), ]
    if rectangle:
        points+=[(.85,.6),(1,1)]

    for ip,p in enumerate(points):
        plt.subplot(1,len(points),ip+1)
        mw=df[['mean','wid']].values
        closest=mw[np.argmin([la.norm(v-p) for v in mw] )]
        pdf=df[(df['mean']==closest[0]) & (df['wid']==closest[1])]
        Js=np.concatenate(pdf['Jaccard'].values)
        plt.hist(Js,bins=np.linspace(0,1,10)),plt.title('Mn {} Wd {}'.format(*closest) )
    plt.savefig(path+'hist.pdf'.format(*closest))

    for ip, p in enumerate(points):
        mw=df[['mean','wid']].values
        closest=mw[np.argmin([la.norm(v-p) for v in mw] )]
        print closest
        pdf=df[(df['mean']==closest[0]) & (df['wid']==closest[1]) ] # &(df['sys']==0)
        detailed_plots(path, df=pdf,save=1,fpath=path,title='Mn {} Wd {}'.format(*closest),fname='Mn_{}_Wd_{}'.format(*closest),format='.pdf' )
    if not hold:
        plt.show()

if __name__=='__main__':
    #NOTES:

    sysargv=['sym1']+list(sys.argv) #Command-line options,

    # DEFAULT OPTIONS
    default={'resolution':11, # Resolution along x-axis in sampling triangle of interaction mean and sd
             'S':50, #Number of species
             'length':100,    # Number of positions along gradient
             'sym':1,
             'systems':range(1), #Replicas (list of labels e.g. ['a','b','c'] or range(3) for 3 replicas)
             'init': 'hysteresis', # Initial conditions
                    # ('uniform' for the same everywhere, 'random' for random,
                    #  'follow' to follow an eq, 'hysteresis' to follow forward then backward)
             'keep_sys':1   # Keep same basic matrix throughout triangle for each replica (seems to give smoother visuals)
             }

    # SETS OF OPTIONS FOR DIFFERENT SIMULATION RUNS
    runs={'default':{},
          'alpha':{'mode':'alpha','triangle':'rectangle'}, #Change interactions rather than carrying capacities
          'rectangle':{'triangle':'rectangle'}, #Explore half rectangle
          'nokeep':{'keep_sys':0, 'systems':range(5)},  # Generate new properties at every point in the triangle
          'gauss':{'distribution':'normal'},
          }
    for symm in [-1,0,1]:
        runs['sym{}'.format(symm) ]={'symm':symm,'resolution':51,'systems':range(15)}

    run =None
    for i in sysargv:
        if i in runs or '+' in i:
            run=i
    if run is None:
        run='default' #Change this to switch between different default runs (put 'default' for default optons)

    def do(run,with_plots=1):
        options={}
        options.update(default)
        if '+' in run:
            for r in run.split('+'):
                options.update(runs[r] )
        else:
            options.update(runs[run])
        path='gradcomp_'+run.replace('+','_')
        if not 'show' in sysargv:
            loop(path=path, rerun='rerun' in sysargv, remeasure='measure' in sysargv,**options)
        if 'detailed' in sysargv:
            detailed_plots(path, save=1)#,filter='mean==.5 & wid==.5')
        if with_plots:
            setfigpath(Path(path)+''.join([i for i in sysargv if '/' in i]) )  # Path for saving figures
            show(path,mode=ifelse('full' in sysargv,'full', 'dft'),triangle=('triangle' in sysargv),hold=1)
        plt.show()

    if 'multirun' in sysargv:
        for run in ['alpha','rectangle','sym0','sym-1']:
            do(run,with_plots=0)
    else:
        do(run)