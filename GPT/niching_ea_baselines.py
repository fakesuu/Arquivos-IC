from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

Array = np.ndarray


def reflect_bounds(x: Array, lo: Array, hi: Array) -> Array:
    x = np.asarray(x, dtype=float).copy()
    span = hi - lo
    y = (x - lo) % (2.0 * span)
    y = np.where(y <= span, y, 2.0 * span - y)
    return lo + y


class NicheArchive:
    """Decision-space fitness-sharing archive; distance is RMS normalized by bounds."""
    def __init__(self, lower: Array, upper: Array, radius: float = 0.15, max_size: int = 500,
                 pressure: float = 1.0):
        self.lo = np.asarray(lower, float)
        self.hi = np.asarray(upper, float)
        self.span = self.hi - self.lo
        self.radius = radius
        self.max_size = max_size
        self.pressure = pressure
        self.X: list[Array] = []
        self.F: list[float] = []

    def norm(self, X: Array) -> Array:
        return (np.asarray(X) - self.lo) / self.span

    def rms_dist(self, X: Array, Y: Array) -> Array:
        Xn = self.norm(np.atleast_2d(X))
        Yn = self.norm(np.atleast_2d(Y))
        d2 = ((Xn[:, None, :] - Yn[None, :, :]) ** 2).mean(axis=2)
        return np.sqrt(d2)

    def update(self, X: Array, F: Array, max_add: Optional[int] = None) -> None:
        X = np.atleast_2d(X); F = np.asarray(F, float).ravel()
        order = np.argsort(F)
        nadd = len(order) if max_add is None else min(len(order), max_add)
        for idx in order[:nadd]:
            x, f = X[idx].copy(), float(F[idx])
            if not self.X:
                self.X.append(x); self.F.append(f); continue
            d = self.rms_dist(x, np.asarray(self.X))[0]
            j = int(np.argmin(d))
            if d[j] < self.radius:
                if f < self.F[j]:
                    self.X[j], self.F[j] = x, f
            else:
                self.X.append(x); self.F.append(f)
        if len(self.X) > self.max_size:
            keep = np.argsort(np.asarray(self.F))[:self.max_size]
            self.X = [self.X[i] for i in keep]
            self.F = [self.F[i] for i in keep]

    def elites(self) -> tuple[Array, Array]:
        if not self.X:
            return np.empty((0, self.lo.size)), np.empty(0)
        o = np.argsort(self.F)
        return np.asarray([self.X[i] for i in o]), np.asarray([self.F[i] for i in o])

    def select(self, X: Array, F: Array, k: int) -> Array:
        """Greedy rank + fitness-sharing selection. Lower score is better."""
        X = np.atleast_2d(X); F = np.asarray(F, float).ravel(); n = len(F)
        k = min(k, n)
        rank_order = np.argsort(F)
        rank = np.empty(n, int); rank[rank_order] = np.arange(n)
        rnorm = rank / max(1, n - 1)
        chosen = [int(rank_order[0])]
        remaining = np.ones(n, dtype=bool); remaining[chosen[0]] = False
        while len(chosen) < k:
            ids = np.flatnonzero(remaining)
            d = self.rms_dist(X[ids], X[np.asarray(chosen)])
            sharing = np.maximum(0.0, 1.0 - d / self.radius) ** 2
            crowd = sharing.sum(axis=1)
            # Reward separation while staying close to good objective rank.
            score = rnorm[ids] + self.pressure * crowd
            j = int(np.argmin(score))
            pick = int(ids[j])
            chosen.append(pick); remaining[pick] = False
        return np.asarray(chosen, int)


def weighted_mean(X: Array, w: Array) -> Array:
    return np.sum(X * w[:, None], axis=0)


class CMAES:
    """Full-covariance CMA-ES with niche-aware environmental selection."""
    def __init__(self, x0: Array, sigma0: float, lam: int, lower: Array, upper: Array,
                 niche: NicheArchive, rng: np.random.Generator):
        self.mean = np.asarray(x0, float).copy()
        self.sigma = float(sigma0)
        self.lam = int(lam)
        self.lo = np.asarray(lower, float); self.hi = np.asarray(upper, float)
        self.d = self.mean.size; self.rng = rng; self.niche = niche
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum(); self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / self.d) / (self.d + 4 + 2*self.mueff/self.d)
        self.cs = (self.mueff + 2) / (self.d + self.mueff + 5)
        self.c1 = 2 / ((self.d + 1.4142)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2*(self.mueff - 2 + 1/self.mueff)/((self.d+2)**2 + self.mueff))
        self.damps = 1 + 2*max(0, math.sqrt((self.mueff-1)/(self.d+1))-1) + self.cs
        self.ps = np.zeros(self.d); self.pc = np.zeros(self.d)
        self.C = np.eye(self.d); self.B = np.eye(self.d); self.D = np.ones(self.d)
        self.BD = self.B * self.D[None, :]
        self.inv_sqrt_C = np.eye(self.d)
        self.eig_age = 0
        self.chiN = self.d**0.5 * (1 - 1/(4*self.d) + 1/(21*self.d**2))
        self.best_x = self.mean.copy(); self.best_f = math.inf

    def ask(self) -> Array:
        z = self.rng.standard_normal((self.lam, self.d))
        y = z @ self.BD.T
        X = self.mean + self.sigma * y
        return reflect_bounds(X, self.lo, self.hi)

    def tell(self, X: Array, F: Array) -> None:
        F = np.asarray(F, float)
        self.niche.update(X, F, max_add=self.lam)
        idx = self.niche.select(X, F, self.mu)
        old_mean = self.mean.copy()
        # Niche score ordering is induced by select; weights favor earlier entries.
        self.mean = weighted_mean(X[idx], self.w)
        y_w = (self.mean - old_mean) / self.sigma
        # Refresh eigendecomposition after the covariance update using current C.
        vals, B = np.linalg.eigh(self.C)
        vals = np.maximum(vals, 1e-20)
        self.B = B; self.D = np.sqrt(vals); self.BD = self.B * self.D[None, :]
        self.inv_sqrt_C = (self.B / self.D[None, :]) @ self.B.T
        self.ps = (1-self.cs)*self.ps + math.sqrt(self.cs*(2-self.cs)*self.mueff) * (self.inv_sqrt_C @ y_w)
        hsig = float(np.linalg.norm(self.ps) / math.sqrt(1-(1-self.cs)**(2*(self.eig_age+1))) / self.chiN < (1.4 + 2/(self.d+1)))
        self.pc = (1-self.cc)*self.pc + hsig*math.sqrt(self.cc*(2-self.cc)*self.mueff) * y_w
        Y = (X[idx] - old_mean) / self.sigma
        rank_mu = np.zeros_like(self.C)
        for wi, yi in zip(self.w, Y): rank_mu += wi * np.outer(yi, yi)
        self.C = (1-self.c1-self.cmu)*self.C + self.c1*np.outer(self.pc, self.pc) + self.cmu*rank_mu
        self.sigma *= math.exp((self.cs/self.damps) * (np.linalg.norm(self.ps)/self.chiN - 1))
        j = int(np.argmin(F))
        if F[j] < self.best_f:
            self.best_f = float(F[j]); self.best_x = X[j].copy()
        self.eig_age += 1


class BIPOPCMAES:
    """BIPOP-style restarts around full CMA-ES, with a persistent niche archive."""
    def __init__(self, x0: Array, sigma0: float, lower: Array, upper: Array,
                 niche_radius: float = 0.15, seed: int = 0):
        self.lo = np.asarray(lower,float); self.hi=np.asarray(upper,float)
        self.d=self.lo.size; self.rng=np.random.default_rng(seed)
        self.x0=np.asarray(x0,float); self.sigma0=float(sigma0)
        self.niche=NicheArchive(self.lo,self.hi,radius=niche_radius,max_size=1000,pressure=0.8)
        self.restart=0

    def minimize(self, fun: Callable[[Array], float], budget: int) -> tuple[Array,float,NicheArchive]:
        evals=0; best_x=self.x0.copy(); best_f=math.inf
        base=max(4, 4+int(3*math.log(self.d+1)))
        large_k=0
        while evals < budget:
            if self.restart % 2 == 0:
                lam=base*(2**large_k)
                sigma_large=self.sigma0*(2.0**large_k)
                large_k += 1
                mean=self.x0.copy()
                if self.niche.X and self.restart>0:
                    # Restart in a less-crowded niche when possible.
                    E, _ = self.niche.elites(); D=self.niche.rms_dist(E, np.asarray([self.x0]))[:,0]
                    far=np.flatnonzero(D > self.niche.radius)
                    if len(far): mean=E[far[int(np.argmax(D[far]))]].copy()
            else:
                max_small=max(base*2, base*(2**max(0,large_k-1)))
                lam=int(max(4, round(math.exp(self.rng.uniform(math.log(base), math.log(max_small))))) )
                if self.niche.X:
                    E,_=self.niche.elites(); mean=E[self.rng.integers(len(E))].copy()
                    mean += self.rng.normal(0, 0.1*(self.hi-self.lo), size=self.d)
                    mean=reflect_bounds(mean,self.lo,self.hi)
                else: mean=self.x0.copy()
            sigma=(sigma_large if self.restart%2==0 else self.sigma0*float(self.rng.lognormal(0.0,0.5)))
            sigma=max(sigma, 1e-12)
            opt=CMAES(mean,sigma,lam,self.lo,self.hi,self.niche,self.rng)
            maxgen=max(1, min((budget-evals)//lam, 2000))
            for _ in range(maxgen):
                X=opt.ask(); F=np.asarray([fun(x) for x in X],float); evals += len(F)
                opt.tell(X,F)
                if opt.best_f<best_f: best_f=opt.best_f; best_x=opt.best_x.copy()
                if evals>=budget: break
            self.restart += 1
        return best_x,best_f,self.niche


class LMCMAES:
    """Practical limited-memory CMA-ES: diagonal+low-rank covariance action, O(d*m) storage."""
    def __init__(self, x0: Array, sigma0: float, lam: int, lower: Array, upper: Array,
                 niche: NicheArchive, memory: int=10, seed: int=0):
        self.mean=np.asarray(x0,float).copy(); self.sigma=float(sigma0); self.lam=int(lam)
        self.lo=np.asarray(lower,float); self.hi=np.asarray(upper,float); self.d=self.mean.size
        self.rng=np.random.default_rng(seed); self.niche=niche; self.m=memory
        self.mu=self.lam//2
        w=np.log(self.mu+0.5)-np.log(np.arange(1,self.mu+1)); self.w=w/w.sum(); self.mueff=1/np.sum(self.w**2)
        self.cs=(self.mueff+2)/(self.d+self.mueff+5); self.cc=(4+self.mueff/self.d)/(self.d+4+2*self.mueff/self.d)
        self.damps=1+2*max(0,math.sqrt((self.mueff-1)/(self.d+1))-1)+self.cs
        self.ps=np.zeros(self.d); self.pc=np.zeros(self.d); self.hist=[]; self.a=np.zeros(0); self.U=np.empty((self.d,0))
        self.chiN=self.d**0.5*(1-1/(4*self.d)+1/(21*self.d**2)); self.best_x=self.mean.copy(); self.best_f=math.inf

    def _A(self,Z: Array, inverse: bool=False) -> Array:
        if self.U.shape[1]==0: return Z
        coeff=(1/np.sqrt(1+self.a)-1) if inverse else (np.sqrt(1+self.a)-1)
        return Z + (Z @ self.U) * coeff[None,:] @ self.U.T

    def ask(self)->Array:
        Z=self.rng.standard_normal((self.lam,self.d)); Y=self._A(Z,False); return reflect_bounds(self.mean+self.sigma*Y,self.lo,self.hi)

    def tell(self,X:Array,F:Array)->None:
        F=np.asarray(F,float); self.niche.update(X,F,max_add=self.lam)
        idx=self.niche.select(X,F,self.mu); old=self.mean.copy(); self.mean=weighted_mean(X[idx],self.w)
        yw=(self.mean-old)/self.sigma
        inv_y=self._A(yw[None,:],True)[0]
        self.ps=(1-self.cs)*self.ps+math.sqrt(self.cs*(2-self.cs)*self.mueff)*inv_y
        hsig=float(np.linalg.norm(self.ps)/math.sqrt(1-(1-self.cs)**2)/self.chiN < 1.4+2/(self.d+1))
        self.pc=(1-self.cc)*self.pc+hsig*math.sqrt(self.cc*(2-self.cc)*self.mueff)*yw
        # Limited-memory update: keep normalized evolution directions; rebuild an orthonormal basis.
        u=self.pc.copy(); nu=np.linalg.norm(u)
        if nu>1e-12: self.hist.append(u/nu)
        self.hist=self.hist[-self.m:]
        if self.hist:
            M=np.column_stack(self.hist)
            Q,_=np.linalg.qr(M,mode='reduced'); self.U=Q
            # Estimate directional variance from selected steps, damped for stability.
            Y=(X[idx]-old)/self.sigma
            proj=Y@self.U
            var=np.sum(self.w[:,None]*(proj**2),axis=0)
            target=np.clip(var-1.0,-0.8,4.0)
            self.a=0.8*self.a+0.2*target if len(self.a)==len(target) else target
        self.sigma*=math.exp((self.cs/self.damps)*(np.linalg.norm(self.ps)/self.chiN-1))
        j=int(np.argmin(F));
        if F[j]<self.best_f:self.best_f=float(F[j]);self.best_x=X[j].copy()

    def minimize(self,fun:Callable[[Array],float],budget:int)->tuple[Array,float,NicheArchive]:
        evals=0
        while evals<budget:
            X=self.ask();F=np.asarray([fun(x) for x in X],float);evals+=len(F);self.tell(X,F)
        return self.best_x,self.best_f,self.niche


class LSHADE:
    """L-SHADE-style DE with success-history adaptation, archive, linear population reduction, and niching."""
    def __init__(self, lower:Array, upper:Array, max_pop:int=100, min_pop:int=4, H:int=10,
                 niche_radius:float=0.15, seed:int=0):
        self.lo=np.asarray(lower,float); self.hi=np.asarray(upper,float); self.d=len(self.lo)
        self.max_pop=max_pop; self.min_pop=min_pop; self.H=H; self.rng=np.random.default_rng(seed)
        self.niche=NicheArchive(self.lo,self.hi,radius=niche_radius,max_size=1000,pressure=0.8)
        self.archive_X=[]; self.archive_F=[]

    def minimize(self,fun:Callable[[Array],float],budget:int,init:Optional[Array]=None)->tuple[Array,float,NicheArchive]:
        if budget < self.min_pop:
            raise ValueError(f"budget must be >= min_pop ({self.min_pop})")
        initial_N=min(self.max_pop, budget)
        N=max(self.min_pop, initial_N)
        P=self.rng.uniform(self.lo,self.hi,size=(N,self.d)) if init is None else np.asarray(init,float).copy()
        if len(P)!=N:
            raise ValueError("init must have shape (population_size, dimension)")
        F=np.asarray([fun(x) for x in P],float); evals=len(F); self.niche.update(P,F,max_add=N)
        MF=np.full(self.H,0.5); MCR=np.full(self.H,0.5); k=0
        max_pop0=N
        while evals < budget and N>=self.min_pop:
            oldP=P.copy(); oldF=F.copy()
            successful_F=[]; successful_CR=[]; successful_dF=[]
            trials=[]; trial_Fs=[]; sampled_F=[]; sampled_CR=[]
            order=np.argsort(F); pcount=max(2,int(math.ceil(0.2*N)))
            pbest_pool=order[:pcount]
            psel=self.niche.select(P[pbest_pool],F[pbest_pool],max(1,min(pcount,pcount//2+1)))
            for i in range(N):
                r=int(self.rng.integers(self.H))
                Fi=None
                for _ in range(50):
                    Fi=float(MF[r]+0.1*self.rng.standard_cauchy())
                    if Fi>0: break
                Fi=float(np.clip(Fi,1e-6,1.0))
                CRi=float(np.clip(MCR[r]+0.1*self.rng.standard_normal(),0,1))
                pbest=int(pbest_pool[int(self.rng.choice(psel))])
                candidates=[j for j in range(N) if j!=i and j!=pbest]
                if not candidates:
                    candidates=[j for j in range(N) if j!=i]
                r1=int(self.rng.choice(candidates))
                pool=np.concatenate([P,np.asarray(self.archive_X)]) if self.archive_X else P
                r2_idx=int(self.rng.integers(len(pool)))
                x2=pool[r2_idx]
                for _ in range(20):
                    if not (np.allclose(x2,P[r1]) or np.allclose(x2,P[pbest]) or np.allclose(x2,P[i])):
                        break
                    r2_idx=int(self.rng.integers(len(pool))); x2=pool[r2_idx]
                vi=P[i]+Fi*(P[pbest]-P[i])+Fi*(P[r1]-x2)
                vi=reflect_bounds(vi,self.lo,self.hi)
                ui=P[i].copy(); jrand=int(self.rng.integers(self.d)); mask=self.rng.random(self.d)<CRi; mask[jrand]=True; ui[mask]=vi[mask]
                trials.append(ui); sampled_F.append(Fi); sampled_CR.append(CRi)
            T=np.asarray(trials); FT=np.asarray([fun(x) for x in T],float); evals+=len(FT)
            self.niche.update(T,FT,max_add=min(len(T),N))
            success=FT < F
            if np.any(success):
                successful_F=np.asarray(sampled_F)[success]
                successful_CR=np.asarray(sampled_CR)[success]
                successful_dF=(F[success]-FT[success])
                ww=successful_dF/(np.sum(successful_dF)+1e-30)
                mf_new=float(np.sum(ww*(successful_F**2))/(np.sum(ww*successful_F)+1e-30))
                mcr_new=float(np.sum(ww*successful_CR))
                MF[k%self.H]=np.clip(mf_new,0,1); MCR[k%self.H]=np.clip(mcr_new,0,1); k+=1
                self.archive_X.extend([oldP[i].copy() for i in np.flatnonzero(success)])
                self.archive_F.extend([float(oldF[i]) for i in np.flatnonzero(success)])
            P[success]=T[success]; F[success]=FT[success]
            if len(self.archive_X)>2*max_pop0:
                oo=np.argsort(self.archive_F)[:2*max_pop0]
                self.archive_X=[self.archive_X[i] for i in oo]; self.archive_F=[self.archive_F[i] for i in oo]
            # Explicit niching: preserve underrepresented archive elites in the live population.
            if self.niche.X and len(self.niche.X)>1:
                E,EF=self.niche.elites(); D=self.niche.rms_dist(E,P); occupied=np.any(D<0.5*self.niche.radius,axis=1)
                missing=np.flatnonzero(~occupied); worst=np.argsort(F)[::-1]; used=0
                for mi in missing:
                    if used>=max(1,N//10): break
                    wi=int(worst[used]); P[wi]=E[mi]; F[wi]=EF[mi]; used+=1
            # Linear population-size reduction from the initial population to min_pop.
            targetN=int(round(max_pop0-(max_pop0-self.min_pop)*min(1.0,evals/max(1,budget))))
            targetN=max(self.min_pop,min(N,targetN))
            if targetN<N:
                keep=self.niche.select(P,F,targetN); P=P[keep]; F=F[keep]; N=targetN
        j=int(np.argmin(F)); return P[j].copy(),float(F[j]),self.niche


# Simple benchmark helpers

def rastrigin(x):
    x=np.asarray(x); return 10*x.size + np.sum(x*x-10*np.cos(2*np.pi*x))


if __name__ == "__main__":
    # Smoke test: all three on bounded 10-D Rastrigin.
    d = 10
    lower, upper = -5.12*np.ones(d), 5.12*np.ones(d)
    f = rastrigin

    bp = BIPOPCMAES(np.zeros(d), 2.5, lower, upper, niche_radius=0.15, seed=1)
    print("BIPOP-CMA-ES:", bp.minimize(f, 300)[:2])

    niche = NicheArchive(lower, upper, radius=0.15, max_size=1000, pressure=0.8)
    lm = LMCMAES(np.zeros(d), 2.5, 24, lower, upper, niche, memory=6, seed=1)
    print("LM-CMA-ES:", lm.minimize(f, 300)[:2])

    ls = LSHADE(lower, upper, max_pop=40, min_pop=6, niche_radius=0.15, seed=1)
    print("L-SHADE:", ls.minimize(f, 300)[:2])
