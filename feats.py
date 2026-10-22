import numpy as np, pandas as pd
def fe_leach(df):
    d=df.sort_values(['sim','node','round']).copy()
    ch=d['is_ch']==1; dr=d['data_recv']; 
    d['fwd_ratio']=np.where(ch,np.where(dr>0,d['bs_recv']/dr.clip(lower=1),-0.5),-1.0)
    d['sent_ratio']=np.where(ch,np.where(dr>0,d['data_to_bs']/(0.5*dr.clip(lower=1)),-0.5),-1.0)
    d['drop_rate']=np.where(d['fwd_ratio']>=0,1-d['fwd_ratio'],-1.0)
    jr=d['join_recv']
    d['data_per_member']=np.where(jr>0,dr/jr.clip(lower=1),0.0)
    d['delivery_ratio']=np.where(jr>0,dr/(5*jr.clip(lower=1)),0.0)
    d['sch_join_ratio']=np.where(jr>0,d['sch_sent']/jr.clip(lower=1),0.0)
    d['join_adv_ratio']=jr/(1+d['adv_recv'])
    g=d.groupby(['sim','node'],sort=False)
    d['_fwd']=d['fwd_ratio'].where(ch)
    d['ema_fwd_ratio']=g['_fwd'].transform(lambda s:s.ewm(alpha=0.125,adjust=False,ignore_na=True).mean()).fillna(-1.0)
    d['ema_adv_sent']=g['adv_sent'].transform(lambda s:s.ewm(alpha=0.125,adjust=False).mean())
    d['_dr']=d['data_recv'].where(ch)
    d['ema_data_recv']=g['_dr'].transform(lambda s:s.ewm(alpha=0.125,adjust=False,ignore_na=True).mean()).fillna(0.0)
    d['ch_freq10']=g['is_ch'].transform(lambda s:s.rolling(10,min_periods=1).mean())
    d['ema_ch_changed']=g['ch_changed'].transform(lambda s:s.ewm(alpha=0.125,adjust=False).mean())
    d['energy_slope']=g['energy_mj'].transform(lambda s:s.rolling(5,min_periods=1).mean())-d['energy_mj']
    low=((d['fwd_ratio']>=0)&(d['fwd_ratio']<0.3)).astype(int); d['_low']=low
    def run(s):
        out=np.zeros(len(s)); c=0
        for i,v in enumerate(s.values):
            c=c+1 if v else 0; out[i]=c
        return pd.Series(out,index=s.index)
    d['anomaly_run']=g['_low'].transform(run)
    def since(s):
        out=np.zeros(len(s)); c=0
        for i,v in enumerate(s.values):
            c=0 if v else c+1; out[i]=c
        return pd.Series(out,index=s.index)
    d['rounds_since_ch']=g['is_ch'].transform(since)
    return d.drop(columns=['_fwd','_dr','_low'])
LEACH_GROUPS={'Statistical flow':['data_sent','data_recv','data_to_bs','bs_recv','fwd_ratio','sent_ratio','drop_rate','data_per_member','delivery_ratio','energy_mj'],
 'Temporal':['ema_fwd_ratio','ema_adv_sent','ema_data_recv','ch_freq10','rounds_since_ch','energy_slope','anomaly_run'],
 'Protocol':['is_ch','adv_sent','adv_recv','join_sent','join_recv','sch_sent','sch_recv','slot_rank','sch_join_ratio','join_adv_ratio'],
 'Topology':['dist_link','dist_to_bs','hop','n_neighbors','n_ch_heard','ch_changed','lqi','lqi_dev','ema_ch_changed']}
LEACH_FEATS=[c for g in LEACH_GROUPS.values() for c in g]
