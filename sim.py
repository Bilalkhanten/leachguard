"""LEACH-Sim: a small LEACH WSN simulator with four attacker types.
Writes leach_sim.csv: one row per (simulation, round, node)."""
import numpy as np, pandas as pd, sys
CLS=['Normal','Blackhole','Grayhole','Flooding','Scheduling']
def simulate(seed,N=100,R=60,area=100.0,bs=(50.0,175.0),p=0.05,rng_radio=45.0):
    rng=np.random.default_rng(seed)
    pos=rng.uniform(0,area,(N,2)); bsx=np.array(bs)
    d_bs=np.linalg.norm(pos-bsx,axis=1)
    frac=rng.uniform(0.05,0.25); nm=int(round(frac*N)); mal=rng.choice(N,nm,replace=False)
    mtype=np.zeros(N,int); mtype[mal]=rng.integers(1,5,nm)
    gh=rng.uniform(0.1,0.7,N)
    D=np.linalg.norm(pos[:,None]-pos[None],axis=2)
    n_nb=(D<30).sum(1)-1
    energy=np.full(N,2.0); lastCH=np.full(N,-100); prev=np.full(N,-1)
    rows=[]
    for r in range(R):
        alive=energy>0
        el=alive&(r-lastCH>=int(1/p))
        T=p/(1-p*(r%int(1/p)))
        isch=(rng.random(N)<T)&el&(mtype==0)
        isch|=(rng.random(N)<0.45)&alive&(mtype>0)
        if not isch.any():
            c=rng.choice(np.where(alive&(mtype==0))[0]); isch[c]=True
        lastCH[isch&(mtype==0)]=r
        ch=np.where(isch)[0]
        advS=np.zeros(N); advS[ch]=1
        fl=ch[mtype[ch]==3]; advS[fl]=rng.integers(3,40,len(fl))
        heard=(D[:,ch]<rng_radio)&(rng.random((N,len(ch)))<0.95)&(ch[None,:]!=np.arange(N)[:,None])
        advR=(heard*advS[ch][None,:]).sum(1)
        nch_heard=heard.sum(1)
        rssi=-D[:,ch]+rng.normal(0,4,(N,len(ch))); rssi[~heard]=-1e9
        pick=np.where(heard.any(1),rssi.argmax(1),-1)
        member=(~isch)&alive&(pick>=0)
        direct=(~isch)&alive&(pick<0)
        chid=np.where(member,ch[np.clip(pick,0,None)],-1)
        joinS=member.astype(float)
        joinR=np.zeros(N); 
        for k,c in enumerate(ch):
            joinR[c]=((chid==c)&(rng.random(N)<0.97)).sum()
        schS=joinR.copy()
        schR=np.where(member,(rng.random(N)<0.97).astype(float),0.0)
        # per-CH interference state (benign noise): short loss bursts
        burst=np.zeros(N); b=isch&(rng.random(N)<0.15); burst[b]=rng.uniform(0.05,0.60,b.sum())
        dup=np.zeros(N); sc=isch&(mtype==4); dup[sc]=rng.uniform(0.2,0.95,sc.sum())
        dataS=np.where(member|direct,rng.integers(3,8,N),0).astype(float)
        dist_mem=np.where(member,D[np.arange(N),np.clip(chid,0,None)],0.0)
        q=np.clip(0.97-0.004*dist_mem,0.5,1)
        dataR=np.zeros(N); rank=np.zeros(N); lqi=np.zeros(N)
        for c in ch:
            ms=np.where(chid==c)[0]
            rank[ms]=np.arange(1,len(ms)+1) if dup[c]==0 else rng.integers(1,max(2,int(len(ms)*(1-dup[c]))+1),len(ms))
            qq=q[ms]*(1-burst[c])*(1-dup[c]*rng.uniform(0.7,1.0))
            dataR[c]=rng.binomial(dataS[ms].astype(int),np.clip(qq,0,1)).sum() if len(ms) else 0
            lqi[c]=np.mean(110-0.6*D[ms,c]+rng.normal(0,3,len(ms))) if len(ms) else 0
        lqi[member]=110-0.6*dist_mem[member]+rng.normal(0,3,member.sum())
        exp_lqi=np.where(member,110-0.6*dist_mem,0.0)
        agg=np.round(dataR*0.5)  # CH aggregation
        pf=np.clip(0.98-0.0015*d_bs-burst,0.3,1); fail=isch&(mtype==0)&(rng.random(N)<0.03); pf[fail]=0.0
        active=rng.random(N)<0.8  # blackhole/grayhole attackers act in 80% of their CH rounds
        recv=np.zeros(N); sent=np.zeros(N)
        for c in ch:
            t=mtype[c]; a=agg[c]
            if t==1: s=0 if active[c] else a
            elif t==2: s=rng.binomial(int(a),1-gh[c]) if active[c] else a
            else: s=a
            sent[c]=s; recv[c]=rng.binomial(int(s),pf[c])
        # flooding nodes also send junk to the BS
        junk=np.zeros(N); junk[fl]=rng.integers(0,30,len(fl)); sent[fl]+=junk[fl]; recv[fl]+=rng.binomial(junk[fl].astype(int),pf[fl])
        # energy (mJ): first-order radio model
        Eel=50e-9; eamp=100e-12; bits=4000; 
        e=np.zeros(N)
        e[member|direct]=dataS[member|direct]*bits*(Eel+eamp*np.where(member,dist_mem,d_bs)[member|direct]**2)
        for c in ch:
            e[c]=(dataR[c]*bits*Eel+agg[c]*bits*(Eel+eamp*d_bs[c]**2)+advS[c]*bits*Eel*8+junk[c]*bits*(Eel+eamp*d_bs[c]**2))
        e*=1e3; energy-= e/1e3; e=e+rng.normal(0,0.005,N)
        for arr in (dataR,sent,recv):
            m_=isch; arr[m_]=np.maximum(0,arr[m_]+np.round(rng.normal(0,1.0,m_.sum())))
        hop=np.where(member,2,np.where(direct,1,0)); hop[isch]=1
        chch=(member&(chid!=prev)).astype(float); prev=np.where(member,chid,-1)
        lab=np.where(isch&(mtype>0),mtype,0)
        for i in np.where(alive)[0]:
            rows.append((seed,r,i,int(isch[i]),D[i,chid[i]] if member[i] else d_bs[i],advS[i],advR[i],joinS[i],joinR[i],schS[i],schR[i],rank[i],
                         dataS[i],dataR[i],sent[i],recv[i],d_bs[i],lqi[i],lqi[i]-exp_lqi[i] if member[i] else 0.0,e[i],hop[i],n_nb[i],nch_heard[i],chch[i],lab[i],frac,mtype[i]))
    cols=['sim','round','node','is_ch','dist_link','adv_sent','adv_recv','join_sent','join_recv','sch_sent','sch_recv','slot_rank','data_sent','data_recv','data_to_bs','bs_recv','dist_to_bs','lqi','lqi_dev','energy_mj','hop','n_neighbors','n_ch_heard','ch_changed','label','mal_frac','node_type']
    return pd.DataFrame(rows,columns=cols)
if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 60
    df=pd.concat([simulate(1000+s) for s in range(n)],ignore_index=True)
    df['label_name']=df['label'].map(dict(enumerate(CLS)))
    df.to_csv('leach_sim.csv',index=False)
    print(df.shape); print(df['label_name'].value_counts()); print(df.groupby('label_name')[['adv_sent','data_recv','bs_recv','data_to_bs']].mean().round(2))
