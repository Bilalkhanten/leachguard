import os
if len(__import__('sys').argv)>1 and __import__('sys').argv[1]=='leach' and not os.path.exists('leach_sim.csv'): os.system('python3 sim.py 80')   # deterministic: regenerates the exact dataset (under a minute)
import sys, json, time, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, roc_auc_score, average_precision_score, f1_score, roc_curve, precision_recall_curve)
import xgboost as xgb, torch, torch.nn as nn, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(2); SEED=42; rng=np.random.RandomState(SEED); torch.manual_seed(SEED)
name=sys.argv[1]; D='data/'
def mf1(y,p,labs=None): return f1_score(y,p,average='macro',labels=labs,zero_division=0)
# ---------------- loaders ----------------
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
def load_leach():
    df=pd.read_csv('leach_sim.csv'); df=fe_leach(df)
    sims=sorted(df['sim'].unique()); test_s=sims[-24:]; trs=sims[:-24]; val_s=trs[-8:]; fit_s=trs[:-8]
    df['y']=df['label_name']
    return df,fit_s,val_s,test_s
def load_nsl():
    cols=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']
    tr=pd.read_csv(D+'KDDTrain.txt',names=cols); te=pd.read_csv(D+'KDDTest.txt',names=cols)
    m={'normal':'Normal'}
    for a in 'neptune smurf pod teardrop land back apache2 udpstorm processtable mailbomb worm'.split(): m[a]='DoS'
    for a in 'satan ipsweep nmap portsweep mscan saint'.split(): m[a]='Probe'
    for a in 'guess_passwd ftp_write imap phf multihop warezmaster warezclient spy xlock xsnoop snmpguess snmpgetattack httptunnel sendmail named'.split(): m[a]='R2L'
    for a in 'buffer_overflow loadmodule rootkit perl sqlattack xterm ps'.split(): m[a]='U2R'
    for d in (tr,te): d['y']=d['label'].map(m); d.drop(columns=['label','difficulty'],inplace=True)
    return tr,te,['protocol_type','service','flag']
def load_unsw():
    tr=pd.read_csv(D+'unsw_train.csv',encoding='utf-8-sig'); te=pd.read_csv(D+'unsw_test.csv',encoding='utf-8-sig')
    for d in (tr,te): d['y']=d['attack_cat'].str.strip(); d.drop(columns=['id','attack_cat','label'],inplace=True,errors='ignore')
    return tr,te,['proto','service','state']
R={'dataset':name}
if name=='leach':
    df,fit_s,val_s,test_s=load_leach(); cat=[]
    classes=['Normal','Blackhole','Grayhole','Flooding','Scheduling']; cid={c:i for i,c in enumerate(classes)}
    feats=LEACH_FEATS
    tr=df[df.sim.isin(fit_s)]; va=df[df.sim.isin(val_s)]; te=df[df.sim.isin(test_s)]
    ytr,yva,yte=[d['y'].map(cid).values for d in (tr,va,te)]
    Xtr,Xva,Xte=[d[feats].astype(float).values for d in (tr,va,te)]
    fn=feats; R['n_sims']={'train':len(fit_s),'val':len(val_s),'test':len(test_s)}
    te_meta=te[['sim','round','node','mal_frac']].reset_index(drop=True)
else:
    tr,te,cat=dict(nsl=load_nsl,unsw=load_unsw)[name]()
    classes=sorted(tr['y'].unique()); cid={c:i for i,c in enumerate(classes)}
    ytr_all=tr['y'].map(cid).values; yte=te['y'].map(cid).values
    num=[c for c in tr.columns if c not in cat+['y']]
    ohe=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(tr[cat])
    prep=lambda d:np.hstack([d[num].astype(float).values,ohe.transform(d[cat])])
    Xall,Xte=prep(tr),prep(te); fn=num+list(ohe.get_feature_names_out(cat))
    itr,iva=train_test_split(np.arange(len(ytr_all)),test_size=0.15,stratify=ytr_all,random_state=SEED)
    Xtr,Xva,ytr,yva=Xall[itr],Xall[iva],ytr_all[itr],ytr_all[iva]
K=len(classes); normal_id=cid['Normal']
R.update(classes=classes,n_train_raw=int(len(ytr)),n_val=int(len(yva)),n_test=int(len(yte)),test_support=np.bincount(yte,minlength=K).tolist(),train_support=np.bincount(ytr,minlength=K).tolist(),n_features_raw=len(fn))
# ---------------- feature selection (RFECV) for LEACH ----------------
sel=np.arange(len(fn))
if name=='leach':
    from sklearn.feature_selection import RFECV
    sub=rng.choice(len(Xtr),30000,replace=False)
    rf0=RandomForestClassifier(n_estimators=40,max_depth=12,n_jobs=2,random_state=SEED)
    rfe=RFECV(rf0,step=2,cv=3,scoring='f1_macro',min_features_to_select=10,n_jobs=1).fit(Xtr[sub],ytr[sub])
    sel=np.where(rfe.support_)[0]; R['rfecv']={'kept':len(sel),'of':len(fn),'dropped':[fn[i] for i in range(len(fn)) if i not in set(sel)]}
    print('RFECV kept',len(sel),'of',len(fn),flush=True)
fn_sel=[fn[i] for i in sel]; R['selected_features']=fn_sel
sc=RobustScaler().fit(Xtr[:,sel])
S=lambda X:np.clip(sc.transform(X[:,sel]),-10,10)
Xtr_s,Xva_s,Xte_s=S(Xtr),S(Xva),S(Xte)
from imblearn.over_sampling import SMOTE
cnt=np.bincount(ytr,minlength=K); target={c:int(max(n,min(int(cnt.max()*0.05),5000))) for c,n in enumerate(cnt)}
Xs,Ys=SMOTE(sampling_strategy=target,random_state=SEED,k_neighbors=max(1,min(5,cnt[cnt>0].min()-1))).fit_resample(Xtr_s,ytr)
print(name,classes,'train',Xs.shape,'val',Xva_s.shape,'test',Xte_s.shape,flush=True)
R['n_train_after_smote']=int(len(Ys))
def norm(P): P=np.nan_to_num(P)+1e-9; return P/P.sum(1,keepdims=True)
def pad(P,cls): Q=np.zeros((len(P),K)); Q[:,cls]=P; return Q
def metrics(y,P,labs=None):
    p=P.argmax(1); pr,rc,f1,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0)
    prw,rcw,f1w,_=precision_recall_fscore_support(y,p,average='weighted',zero_division=0)
    cm=confusion_matrix(y,p,labels=range(K)); fp=cm.sum(0)-np.diag(cm); tn=cm.sum()-cm.sum(0)-cm.sum(1)+np.diag(cm)
    try: auc=roc_auc_score(y,P,multi_class='ovr',average='macro',labels=range(K))
    except Exception: auc=float('nan')
    return dict(acc=accuracy_score(y,p),prec_macro=pr,rec_macro=rc,f1_macro=f1,f1_weighted=f1w,prec_weighted=prw,rec_weighted=rcw,fpr_macro=float(np.mean(fp/np.maximum(fp+tn,1))),mcc=matthews_corrcoef(y,p),bacc=balanced_accuracy_score(y,p),roc_auc=auc)
models={}; val={}; test={}; fit_t={}; tuning={}
big=rng.choice(len(Xs),min(len(Xs),40000),replace=False)
# ---- RF
def obj_rf(t):
    m=RandomForestClassifier(n_estimators=t.suggest_int('n_estimators',50,300),max_features=t.suggest_categorical('max_features',['sqrt','log2']),min_samples_leaf=t.suggest_int('min_samples_leaf',1,4),n_jobs=2,random_state=SEED).fit(Xs[big],Ys[big])
    return mf1(yva,m.predict(Xva_s))
st=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED)); st.optimize(obj_rf,n_trials=8); bp=st.best_params; tuning['RF']=bp
t0=time.time(); models['RF']=RandomForestClassifier(n_jobs=2,random_state=SEED,**bp).fit(Xs,Ys); fit_t['RF']=time.time()-t0
# ---- XGB
def obj_x(t):
    m=xgb.XGBClassifier(n_estimators=t.suggest_int('n_estimators',100,400),max_depth=t.suggest_int('max_depth',3,10),learning_rate=t.suggest_float('learning_rate',0.03,0.3,log=True),subsample=0.8,n_jobs=2,tree_method='hist',random_state=SEED).fit(Xs[big],Ys[big])
    return mf1(yva,m.predict(Xva_s))
st=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED)); st.optimize(obj_x,n_trials=8); bx=st.best_params; tuning['XGB']=bx
t0=time.time(); models['XGB']=xgb.XGBClassifier(subsample=0.8,n_jobs=2,tree_method='hist',random_state=SEED,**bx).fit(Xs,Ys); fit_t['XGB']=time.time()-t0
# ---- SVM
ssub=rng.choice(len(Xs),min(len(Xs),20000),replace=False); s10=ssub[:8000]
def obj_s(t):
    m=SVC(C=t.suggest_float('C',0.1,1000,log=True),gamma=t.suggest_float('gamma',1e-4,1.0,log=True)).fit(Xs[s10],Ys[s10]); return mf1(yva[:8000],m.predict(Xva_s[:8000]))
st=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED)); st.optimize(obj_s,n_trials=6); bs=st.best_params; tuning['SVM']=bs
t0=time.time(); models['SVM']=SVC(probability=True,random_state=SEED,**bs).fit(Xs[ssub],Ys[ssub]); fit_t['SVM']=time.time()-t0; R['svm_train_rows']=int(len(ssub))
# ---- kNN
ksub=rng.choice(len(Xs),min(len(Xs),60000),replace=False); best=(-1,None)
for k in (3,5,7,11,15):
    m=KNeighborsClassifier(n_neighbors=k,weights='distance',n_jobs=2).fit(Xs[ksub],Ys[ksub]); sc_=mf1(yva[:8000],m.predict(Xva_s[:8000]))
    if sc_>best[0]: best=(sc_,k)
tuning['kNN']={'k':best[1]}
t0=time.time(); models['kNN']=KNeighborsClassifier(n_neighbors=best[1],weights='distance',n_jobs=2).fit(Xs[ksub],Ys[ksub]); fit_t['kNN']=time.time()-t0
# ---- LSTM
F=Xs.shape[1]; steps=8; width=int(np.ceil(F/steps)); padn=steps*width-F
def seq(X): return torch.tensor(np.pad(X,((0,0),(0,padn))).reshape(len(X),steps,width),dtype=torch.float32)
class Net(nn.Module):
    def __init__(s,u,dp): super().__init__(); s.l=nn.LSTM(width,u,batch_first=True); s.f=nn.Sequential(nn.Dropout(dp),nn.Linear(u,K))
    def forward(s,x): o,_=s.l(x); return s.f(o[:,-1])
lossf=nn.CrossEntropyLoss()
def train_lstm(u,dp,X,Y,epochs,patience=4,log=False):
    torch.manual_seed(SEED); net=Net(u,dp); opt=torch.optim.Adam(net.parameters(),1e-3); Xt,Yt=seq(X),torch.tensor(Y); Xv=seq(Xva_s); Yv=torch.tensor(yva)
    best=(1e9,None); bad=0; h={'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
    for ep in range(epochs):
        net.train(); perm=torch.randperm(len(Xt)); tl=0;tc=0
        for i in range(0,len(Xt),512):
            b=perm[i:i+512]; opt.zero_grad(); o=net(Xt[b]); l=lossf(o,Yt[b]); l.backward(); opt.step(); tl+=l.item()*len(b); tc+=(o.argmax(1)==Yt[b]).sum().item()
        net.eval()
        with torch.no_grad(): ov=net(Xv); vl=lossf(ov,Yv).item(); va_=(ov.argmax(1)==Yv).float().mean().item()
        h['train_loss'].append(tl/len(Xt)); h['train_acc'].append(tc/len(Xt)); h['val_loss'].append(vl); h['val_acc'].append(va_)
        if vl<best[0]-1e-4: best=(vl,{k:v.clone() for k,v in net.state_dict().items()},ep+1); bad=0
        else:
            bad+=1
            if bad>=patience: break
    net.load_state_dict(best[1]); net.eval(); return net,best[0],best[2],h
cands=[(32,0.1),(64,0.3),(128,0.3),(128,0.5)]; res=[]
for u,dp in cands:
    _,vl,_,_=train_lstm(u,dp,Xs[big],Ys[big],6); res.append((vl,u,dp))
_,u,dp=min(res); tuning['LSTM']={'hidden_units':u,'dropout':dp,'steps':steps}
t0=time.time(); net,vl,bep,hist=train_lstm(u,dp,Xs,Ys,40); fit_t['LSTM']=time.time()-t0
R['lstm_best_epoch']=bep; R['lstm_epochs_run']=len(hist['val_loss']); R['lstm_hist']=hist; lstm=net
def lstm_proba(X):
    with torch.no_grad(): return torch.softmax(lstm(seq(X)),1).numpy()
def proba(nm,X):
    if nm=='LSTM': return lstm_proba(X)
    m=models[nm]; return pad(m.predict_proba(X),m.classes_)
names=['RF','SVM','XGB','LSTM','kNN']; mem=names
for nm in names: val[nm]=norm(proba(nm,Xva_s)); test[nm]=norm(proba(nm,Xte_s))
def ens(P,w): return sum(w[i]*P[m] for i,m in enumerate(mem))
def obj_w(t):
    w=np.array([t.suggest_float(f'w{i}',0,1) for i in range(len(mem))]); 
    return 0 if w.sum()==0 else mf1(yva,ens(val,w/w.sum()).argmax(1))
st=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED)); st.optimize(obj_w,n_trials=80)
w=np.array([st.best_params[f'w{i}'] for i in range(len(mem))]); w/=w.sum(); R['ensemble_weights']=dict(zip(mem,w.tolist()))
test['Ensemble']=ens(test,w); val['Ensemble']=ens(val,w)
# ---- anomaly detectors (binary, trained on normal training rows only)
ybin=(yte!=normal_id).astype(int); Xn=Xtr_s[ytr==normal_id]; Xn=Xn[rng.choice(len(Xn),min(len(Xn),100000),replace=False)]
vb=(yva!=normal_id).astype(int)
iso=IsolationForest(n_estimators=200,random_state=SEED,n_jobs=2).fit(Xn)
R['iforest']={'auc':float(roc_auc_score(ybin,-iso.score_samples(Xte_s))),'ap':float(average_precision_score(ybin,-iso.score_samples(Xte_s)))}
class AE(nn.Module):
    def __init__(s,b): super().__init__(); s.e=nn.Sequential(nn.Linear(F,max(2*b,16)),nn.ReLU(),nn.Linear(max(2*b,16),b),nn.ReLU()); s.d=nn.Sequential(nn.Linear(b,max(2*b,16)),nn.ReLU(),nn.Linear(max(2*b,16),F))
    def forward(s,x): return s.d(s.e(x))
def train_ae(b,epochs=15):
    torch.manual_seed(SEED); m=AE(b); o=torch.optim.Adam(m.parameters(),1e-3); X=torch.tensor(Xn,dtype=torch.float32)
    for ep in range(epochs):
        perm=torch.randperm(len(X))
        for i in range(0,len(X),512):
            bb=X[perm[i:i+512]]; o.zero_grad(); l=((m(bb)-bb)**2).mean(); l.backward(); o.step()
    return m.eval()
def ae_score(m,X):
    with torch.no_grad(): x=torch.tensor(X,dtype=torch.float32); return ((m(x)-x)**2).mean(1).numpy()
best=(-1,None,None)
for b in (4,8,16):
    m=train_ae(b); a=roc_auc_score(vb,ae_score(m,Xva_s))
    if a>best[0]: best=(a,b,m)
tuning['AE']={'bottleneck':best[1]}; ae=best[2]
R['autoencoder']={'auc':float(roc_auc_score(ybin,ae_score(ae,Xte_s))),'ap':float(average_precision_score(ybin,ae_score(ae,Xte_s)))}
tuning['IF']={'n_estimators':200}
R['tuning']=tuning
R['binary_auc']={nm:float(roc_auc_score(ybin,1-test[nm][:,normal_id])) for nm in test}; R['binary_ap']={nm:float(average_precision_score(ybin,1-test[nm][:,normal_id])) for nm in test}
# ---- evaluation
R['test_metrics']={nm:metrics(yte,test[nm]) for nm in names+['Ensemble']}
for nm in R['test_metrics']: R['test_metrics'][nm]['train_s']=fit_t.get(nm)
pe=test['Ensemble'].argmax(1); cm=confusion_matrix(yte,pe,labels=range(K)); R['confusion_matrix']=cm.tolist()
pr,rc,f1,sp=precision_recall_fscore_support(yte,pe,zero_division=0,labels=range(K))
R['per_class']={classes[i]:dict(prec=pr[i],rec=rc[i],f1=f1[i],support=int(sp[i]),fpr=float((cm[:,i].sum()-cm[i,i])/max(cm.sum()-cm[i].sum(),1)),fnr=float(1-rc[i]),aucpr=float(average_precision_score((yte==i).astype(int),test['Ensemble'][:,i])),aucroc=float(roc_auc_score((yte==i).astype(int),test['Ensemble'][:,i]))) for i in range(K)}
models['RF'].set_params(n_jobs=1); models['kNN'].set_params(n_jobs=1); models['XGB'].set_params(n_jobs=1); torch.set_num_threads(1)
# ---- resource cost (this CPU), model size, single-sample latency, throughput
def size_kb(nm):
    if nm=='LSTM': return sum(p.numel() for p in lstm.parameters())*4/1024
    if nm=='XGB': return len(models['XGB'].get_booster().save_raw())/1024
    return len(pickle.dumps(models[nm]))/1024
cost={}
x1=Xte_s[:300]
for nm in names:
    lat=[]
    for i in range(200):
        t0=time.perf_counter(); proba(nm,x1[i:i+1]); lat.append((time.perf_counter()-t0)*1000)
    xb=Xte_s[:2000]; t0=time.perf_counter(); proba(nm,xb); thr=len(xb)/(time.perf_counter()-t0)
    cost[nm]={'latency_ms':float(np.median(lat)),'throughput_sps':float(thr),'model_kb':float(size_kb(nm))}
lat=[]
for i in range(200):
    t0=time.perf_counter(); sum(w[j]*proba(m,x1[i:i+1]) for j,m in enumerate(mem)); lat.append((time.perf_counter()-t0)*1000)
t0=time.perf_counter(); sum(w[j]*proba(m,Xte_s[:2000]) for j,m in enumerate(mem)); thr=2000/(time.perf_counter()-t0)
cost['Ensemble']={'latency_ms':float(np.median(lat)),'throughput_sps':float(thr),'model_kb':float(sum(cost[m]['model_kb'] for m in mem))}
lat=[]
for i in range(200):
    t0=time.perf_counter(); iso.score_samples(x1[i:i+1]); lat.append((time.perf_counter()-t0)*1000)
cost['IF']={'latency_ms':float(np.median(lat)),'model_kb':len(pickle.dumps(iso))/1024}
lat=[]
for i in range(200):
    t0=time.perf_counter(); ae_score(ae,x1[i:i+1]); lat.append((time.perf_counter()-t0)*1000)
cost['AE']={'latency_ms':float(np.median(lat)),'model_kb':sum(p.numel() for p in ae.parameters())*4/1024}
R['cost']=cost
# ---- adversarial (feature-space, scaled units): white-box on LSTM, transferred to the rest
def attack(X,y,eps,pgd=False,n=10):
    x0=seq(X); yy=torch.tensor(y); x=x0.clone(); a=eps/4 if pgd else eps
    for _ in range(n if pgd else 1):
        x.requires_grad_(True); l=lossf(lstm(x),yy); g,=torch.autograd.grad(l,x); x=(x+a*g.sign()).detach(); x=x0+torch.clamp(x-x0,-eps,eps)
    return x.reshape(len(X),-1)[:,:F].numpy()
idx=np.concatenate([rng.choice(np.where(yte==c)[0],min(400,(yte==c).sum()),replace=False) for c in range(K)]); Xa,ya=Xte_s[idx],yte[idx]; adv={}; R['adv_n']=int(len(idx))
for kind in ('FGSM','PGD'):
    for eps in (0.01,0.05,0.10,0.15):
        Xadv=attack(Xa,ya,eps,pgd=(kind=='PGD')); r={}
        for nm in names: r[nm]=float(accuracy_score(ya,proba(nm,Xadv).argmax(1)))
        r['Ensemble']=float(accuracy_score(ya,sum(w[i]*norm(proba(m,Xadv)) for i,m in enumerate(mem)).argmax(1))); adv[f'{kind}_{eps}']=r
R['adv']=adv; R['adv_clean']={nm:float(accuracy_score(ya,test[nm][idx].argmax(1))) for nm in names+['Ensemble']}
R['adv_clean_f1']={nm:float(mf1(ya,test[nm][idx].argmax(1))) for nm in names+['Ensemble']}
# ---- SHAP (XGB) + importance agreement
import shap
ss=rng.choice(len(Xte_s),min(2000,len(Xte_s)),replace=False)
sv_=np.array(shap.TreeExplainer(models['XGB']).shap_values(Xte_s[ss])); fa=[a for a,n_ in enumerate(sv_.shape) if n_==Xte_s.shape[1]][-1]
imp=np.abs(sv_).mean(axis=tuple(i for i in range(sv_.ndim) if i!=fa)); R['shap_all']=[(fn_sel[i],float(imp[i]),float(imp[i]/imp.sum())) for i in np.argsort(-imp)]
from scipy.stats import spearmanr
rfimp=models['RF'].feature_importances_
R['shap_vs_rf_spearman']=float(spearmanr(imp,rfimp).correlation)
rk=lambda v:np.argsort(np.argsort(-v))+1
R['rank_table']=[(fn_sel[i],int(rk(imp)[i]),int(rk(rfimp)[i])) for i in np.argsort(-imp)]
half=rng.permutation(len(Xte_s)); h1,h2=half[:len(half)//2][:1500],half[len(half)//2:][:1500]
def shap_imp(ix):
    s=np.array(shap.TreeExplainer(models['XGB']).shap_values(Xte_s[ix])); f=[a for a,n_ in enumerate(s.shape) if n_==Xte_s.shape[1]][-1]; return np.abs(s).mean(axis=tuple(i for i in range(s.ndim) if i!=f))
i1,i2=shap_imp(h1),shap_imp(h2); R['shap_split_half_spearman']=float(spearmanr(i1,i2).correlation)
if name=='leach':
    grp={f:g for g,fs in LEACH_GROUPS.items() for f in fs}; R['feature_group']={f:grp[f] for f in fn_sel}
    gs={}
    for f,_,s in R['shap_all']: gs[grp[f]]=gs.get(grp[f],0)+s
    R['shap_group_share']=gs; R['group_counts']={g:sum(1 for f in fn_sel if grp[f]==g) for g in LEACH_GROUPS}
    R['mi']={}
    from sklearn.feature_selection import mutual_info_classif
    mi=mutual_info_classif(Xtr_s[big],ytr[big] if len(ytr)==len(Xtr_s) else None,random_state=SEED) if False else mutual_info_classif(Xtr_s[rng.choice(len(Xtr_s),30000,replace=False)],ytr[:0] if False else ytr[np.sort(rng.choice(len(Xtr_s),30000,replace=False))] if False else None) if False else None
    ix=rng.choice(len(Xtr_s),30000,replace=False); mi=mutual_info_classif(Xtr_s[ix],ytr[ix],random_state=SEED); R['mi']={fn_sel[i]:float(mi[i]) for i in range(len(fn_sel))}
    # --- per-simulation stability, malicious-fraction bins
    pe=test['Ensemble'].argmax(1); per=[]
    for s_,g in te_meta.groupby('sim'):
        ii=g.index.values; labs=sorted(set(yte[ii])|set(pe[ii])); per.append((float(g['mal_frac'].iloc[0]),float(accuracy_score(yte[ii],pe[ii])),float(mf1(yte[ii],pe[ii],labs))))
    per=np.array(per); R['per_sim']={'acc_mean':float(per[:,1].mean()),'acc_std':float(per[:,1].std()),'f1_mean':float(per[:,2].mean()),'f1_std':float(per[:,2].std()),'n':len(per)}
    bins=np.quantile(te_meta['mal_frac'].unique(),[0,.2,.4,.6,.8,1.0]); bins[-1]+=1e-9; frac_res={}
    for nm in names+['Ensemble']:
        out=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            m=(te_meta['mal_frac']>=lo)&(te_meta['mal_frac']<hi); ii=np.where(m)[0]; out.append((float((lo+hi)/2*100),float(mf1(yte[ii],test[nm][ii].argmax(1)))))
        frac_res[nm]=out
    R['malfrac_sweep']=frac_res
    # --- extra-channel-loss noise sweep recomputed through the feature pipeline
    nz={}; raw_te=df[df.sim.isin(test_s)].drop(columns=[c for c in df.columns if c in set(fn)-set(['dist_link','dist_to_bs','hop','n_neighbors','n_ch_heard','ch_changed','lqi','lqi_dev','is_ch','adv_sent','adv_recv','join_sent','join_recv','sch_sent','sch_recv','slot_rank','data_sent','data_recv','data_to_bs','bs_recv','energy_mj'])])
    raw_te=raw_te.reset_index(drop=True); nr=np.random.default_rng(7)
    for pl in (0,5,10,15,20,25):
        d=raw_te.copy()
        for c in ('data_recv','bs_recv','join_recv','sch_recv'): d[c]=nr.binomial(d[c].astype(int),1-pl/100)
        d=fe_leach(d)
        # restore original row order by (sim,node,round) to match yte order
        d=d.set_index(['sim','node','round']).loc[list(zip(te['sim'],te['node'],te['round']))].reset_index()
        Xn_=S(d[feats].astype(float).values); P={nm:norm(proba(nm,Xn_)) for nm in names}; P['Ensemble']=ens(P,w)
        nz[pl]={nm:float(mf1(yte,P[nm].argmax(1))) for nm in P}
    R['noise_sweep']=nz
np.savez_compressed(f'arrays_{name}.npz',yte=yte,**{f'p_{k}':v.astype(np.float32) for k,v in test.items()},ybin=ybin,if_score=-iso.score_samples(Xte_s),ae_score=ae_score(ae,Xte_s))
json.dump(R,open(f'results2_{name}.json','w'),indent=1,default=float)
print('done',name,{k:round(v['acc'],4) for k,v in R['test_metrics'].items()},flush=True)
