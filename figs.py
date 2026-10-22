import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':7.5,'axes.linewidth':0.6,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':400,'savefig.dpi':400})
C={'RF':'#1b7837','SVM':'#762a83','XGB':'#e08214','LSTM':'#2166ac','kNN':'#8c510a','Ensemble':'#b2182b'}
L=json.load(open('results2_leach.json')); N=json.load(open('results2_nsl.json')); U=json.load(open('results2_unsw.json'))
# Compact score files (scores_*.npz) ship with the package; the full probability arrays (arrays_*.npz) are written by run2.py and are used when present.
def _score(name,m,k):
    f=f'arrays_{name}.npz'
    if os.path.exists(f): return 1-np.load(f)[f'p_{m}'][:,k]
    return np.load(f'scores_{name}.npz')[f'score_{m}']
def _yte(name): return np.load(f'arrays_{name}.npz' if os.path.exists(f'arrays_{name}.npz') else f'scores_{name}.npz')['yte']
def _unsw_pred():
    f='arrays_unsw.npz'
    return np.load(f)['p_Ensemble'].argmax(1) if os.path.exists(f) else np.load('scores_unsw.npz')['pred_Ensemble'].astype(int)
SZ={'fig2':(4.92,7.29),'fig3':(6.64,4.27),'fig4':(5.65,1.92),'fig5':(5.86,2.78),'fig6':(5.17,2.59),'fig7':(4.32,3.01),'fig8':(4.47,2.91),'fig9':(4.02,2.79),'fig10':(4.23,2.67),'fig11':(6.05,3.18)}
def box(ax,x,y,w,h,t,fc='#eef3fb',ec='#2b4a7a',fs=7.5,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=0.012',fc=fc,ec=ec,lw=0.8)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',wrap=True)
def arr(ax,x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=8,lw=0.8,color='#333'))
def line(ax,x1,y1,x2,y2): ax.plot([x1,x2],[y1,y2],color='#333',lw=0.8,solid_capstyle='butt')
def newfig(k): 
    f=plt.figure(figsize=SZ[k]); ax=f.add_axes([0,0,1,1]); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1); return f,ax
def pbox(ax,x,y,w,h,t,fc='#eef3fb',ec='#2b4a7a',fs=7.5,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.01',fc=fc,ec=ec,lw=0.9,zorder=3))
    if t: ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',zorder=4)
def A(ax,x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=0.9,color='#222',shrinkA=0,shrinkB=0,zorder=5))
def Ln(ax,*pts):
    ax.plot([p[0] for p in pts],[p[1] for p in pts],color='#222',lw=0.9,solid_capstyle='butt',solid_joinstyle='miter',zorder=5)
def newfig(k):
    f=plt.figure(figsize=SZ[k]); ax=f.add_axes([0,0,1,1]); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1); return f,ax
def phase(ax,y0,y1,t):
    ax.add_patch(FancyBboxPatch((0.03,y0),0.94,y1-y0,boxstyle='round,pad=0,rounding_size=0.01',fc='#fafafa',ec='#9aa',lw=0.7,ls='--',zorder=1))
    ax.text(0.05,y1-0.012,t,va='top',fontsize=7.5,fontweight='bold',color='#333',zorder=2)
# ---- Fig 2 workflow
f,ax=newfig('fig2')
phase(ax,0.805,0.985,'Phase 1: data and features')
for x,t in ((0.06,'LEACH-Sim\n(own simulation)'),(0.37,'NSL-KDD\n(official split)'),(0.68,'UNSW-NB15\n(official split)')): pbox(ax,x,0.895,0.26,0.05,t,fs=7)
pbox(ax,0.07,0.822,0.86,0.034,'Feature engineering (four groups) + RFECV on LEACH-Sim',fs=6.8)
for x in (0.19,0.50,0.81): A(ax,x,0.895,x,0.856)
phase(ax,0.515,0.78,'Phase 2: training')
pbox(ax,0.07,0.683,0.40,0.045,'SMOTE on training part only',fs=7); pbox(ax,0.53,0.683,0.40,0.045,'TPE tuning on validation split',fs=7)
A(ax,0.36,0.822,0.36,0.728); A(ax,0.47,0.7055,0.53,0.7055)
cx=[0.10+0.125*i for i in range(5)]
Ln(ax,(0.73,0.683),(0.73,0.665)); Ln(ax,(cx[0],0.665),(0.825,0.665))
for x,m in zip(cx,['RF','SVM','XGB','LSTM','kNN']):
    pbox(ax,x-0.05,0.58,0.10,0.05,m,fs=7); A(ax,x,0.665,x,0.63)
pbox(ax,0.70,0.555,0.25,0.085,'Isolation forest +\nautoencoder\n(normal data only)',fc='#f4f4f4',fs=6.5); A(ax,0.825,0.665,0.825,0.64)
phase(ax,0.335,0.495,'Phase 3: ensemble')
for x in cx: Ln(ax,(x,0.58),(x,0.548))
Ln(ax,(cx[0],0.548),(cx[-1],0.548)); A(ax,cx[2],0.548,cx[2],0.435)
pbox(ax,0.06,0.37,0.58,0.065,'Soft vote: weights tuned on\nvalidation macro-F1',fc='#fdeaea',ec='#b2182b',fs=7)
A(ax,0.825,0.555,0.825,0.435)
pbox(ax,0.70,0.37,0.25,0.065,'Anomaly scores\n(binary flag)',fc='#f4f4f4',fs=6.8)
phase(ax,0.04,0.31,'Phase 4: evaluation')
A(ax,cx[2],0.37,cx[2],0.31); A(ax,0.825,0.37,0.825,0.31)
pbox(ax,0.07,0.205,0.40,0.045,'Macro-F1, MCC, ROC / PR curves',fs=7); pbox(ax,0.53,0.205,0.40,0.045,'Per-simulation stability, class errors',fs=7)
pbox(ax,0.07,0.135,0.40,0.045,'FGSM / PGD transfer attacks',fs=7); pbox(ax,0.53,0.135,0.40,0.045,'SHAP feature importance',fs=7)
pbox(ax,0.07,0.065,0.86,0.045,'Model size and inference time (x86 CPU measurement)',fs=7)
f.savefig('figs/fig2.png'); plt.close(f)
# ---- Fig 3 deployment
f,ax=newfig('fig3')
pbox(ax,0.16,0.74,0.68,0.19,'Base station / gateway\nfull ensemble (RF, XGB, LSTM, SVM, kNN)',fc='#fdeaea',ec='#b2182b',fs=7.5)
for cxc,i in ((0.30,1),(0.70,2)):
    pbox(ax,cxc-0.17,0.46,0.34,0.15,f'Cluster head {i}\nforwards aggregated data;\noptional light detector',fs=7)
    A(ax,cxc,0.61,cxc,0.74)
    for dx in (-0.115,0,0.115):
        xs=cxc+dx; pbox(ax,xs-0.045,0.17,0.09,0.11,'node',fs=6.5,fc='#f6f6f6'); A(ax,xs,0.28,xs,0.46)
cm=L['cost']
ax.text(0.5,0.085,'Leaf nodes only sense and forward; detection runs where memory and power allow.',ha='center',fontsize=7,style='italic')
ax.text(0.5,0.03,f"Model size on disk (this study): autoencoder {cm['AE']['model_kb']:.0f} KB, LSTM {cm['LSTM']['model_kb']:.0f} KB, XGBoost {cm['XGB']['model_kb']/1024:.1f} MB, full ensemble {cm['Ensemble']['model_kb']/1024:.1f} MB",ha='center',fontsize=6.5,color='#333')
f.savefig('figs/fig3.png'); plt.close(f)
# ---- Fig 4 soft vote
f,ax=newfig('fig4')
w=L['ensemble_weights']; pbox(ax,0.02,0.36,0.13,0.28,'Feature\nvector',fs=7)
ys=[0.86-i*0.17 for i in range(5)]
Ln(ax,(0.15,0.5),(0.21,0.5)); Ln(ax,(0.21,ys[0]),(0.21,ys[-1]))
for i,m in enumerate(['RF','SVM','XGB','LSTM','kNN']):
    y=ys[i]; A(ax,0.21,y,0.30,y); pbox(ax,0.30,y-0.06,0.20,0.12,f'{m}  P(class)',fs=6.8)
    Ln(ax,(0.50,y),(0.66,y)); ax.text(0.58,y+0.02,f'w = {w[m]:.2f}',fontsize=6.3,ha='center',va='bottom',zorder=6)
Ln(ax,(0.66,ys[0]),(0.66,ys[-1])); A(ax,0.66,0.5,0.72,0.5)
pbox(ax,0.72,0.30,0.14,0.40,'Weighted\nmean of\nprobabilities',fc='#fdeaea',ec='#b2182b',fs=6.8)
A(ax,0.86,0.5,0.90,0.5); pbox(ax,0.90,0.40,0.085,0.20,'class',fs=6.8)
f.savefig('figs/fig4.png'); plt.close(f)
# ---- Fig methodology: schematic (lanes)
SZ['figm']=(6.5,4.7); plt.rcParams['mathtext.fontset']='dejavuserif'
f,ax=newfig('figm')
def Ad(ax,x1,y1,x2,y2,c='#222'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,lw=0.9,color=c,shrinkA=0,shrinkB=0,zorder=5,linestyle=(0,(3,2))))
# split bar drawn to scale
tot=268591+45256+134447; x0,x1=0.16,0.985; segs=[('train 60 %',268591,'#cfdcf7','#2b4a7a'),('val 10 %',45256,'#ffe2b0','#c98a22'),('test 30 %',134447,'#f7c6c6','#b2182b')]
xx=x0
for t,n,fc,ec in segs:
    w=(x1-x0)*n/tot; ax.add_patch(plt.Rectangle((xx,0.915),w,0.06,fc=fc,ec=ec,lw=0.9,zorder=3)); ax.text(xx+w/2,0.945,t,ha='center',va='center',fontsize=6.6,zorder=4); xx+=w
ax.text(0.015,0.9425,'448,294 records\n80 networks',ha='left',va='center',fontsize=6.2)
ax.text(0.5,0.885,'split by whole simulation: 48 / 8 / 24 networks',ha='center',va='center',fontsize=6.2,style='italic',color='#444')
cy=[0.72,0.45,0.18]; H=0.13
lanes=[('Train','#eef3fb','#2b4a7a'),('Validation','#fff4e0','#c98a22'),('Test','#fdeaea','#b2182b')]
for y,(nm,fc,ec) in zip(cy,lanes):
    ax.add_patch(FancyBboxPatch((0.12,y-0.1),0.875,0.2,boxstyle='round,pad=0,rounding_size=0.01',fc='#fcfcfd',ec=ec,lw=0.8,ls='--' if nm=='Test' else '-',zorder=1))
    pbox(ax,0.015,y-0.1,0.095,0.2,nm,fc=fc,ec=ec,fs=6.5,bold=True)
cols=[(0.135,0.115),(0.275,0.15),(0.46,0.155),(0.66,0.165),(0.855,0.13)]
def B(c,lane,t,fc,ec='#2b4a7a',fs=6.4): x,w=cols[c]; pbox(ax,x,cy[lane]-H/2,w,H,t,fc=fc,ec=ec,fs=fs)
def mid(c): return cols[c][0]+cols[c][1]/2
def L_(c,lane): return cols[c][0]
def R_(c,lane): return cols[c][0]+cols[c][1]
Bc,Oc,Rc='#eef3fb','#fff4e0','#fdeaea'
B(0,0,'$D_{train}$\n268,591',Bc); B(1,0,'RFECV: 36 → 18\nscaler $S$ fitted',Bc); B(2,0,'SMOTE ($k$ = 5)\n268,591 → 277,073',Bc,fs=6.1); B(3,0,'fit $f_m$\nRF · SVM · XGB\nLSTM · k-NN',Bc); B(4,0,'IF · AE\nnormal rows\nonly',Bc,fs=6.2)
B(0,1,'$D_{val}$\n45,256',Oc,'#c98a22'); B(1,1,'apply $S$, 18 feat.\n(no refit)',Oc,'#c98a22'); B(3,1,'tune $\\theta$ (TPE)\nweights $w$, $\\Sigma w$ = 1\nearly stopping',Oc,'#c98a22')
B(0,2,'$D_{test}$\n134,447',Rc,'#b2182b'); B(1,2,'apply $S$, 18 feat.\n(no refit)',Rc,'#b2182b'); B(3,2,'$\\hat{y}=\\arg\\max\\,\\Sigma_m\\, w_m p_m$',Rc,'#b2182b',fs=6.8); B(4,2,'macro-F1, MCC,\nAUC, FGSM,\nSHAP',Rc,'#b2182b',fs=6.2)
for lane in (0,1,2): A(ax,R_(0,lane),cy[lane],L_(1,lane),cy[lane])
A(ax,R_(1,0),cy[0],L_(2,0),cy[0]); A(ax,R_(2,0),cy[0],L_(3,0),cy[0]); A(ax,R_(3,0),cy[0],L_(4,0),cy[0])
for lane in (1,2):
    A(ax,R_(1,lane),cy[lane],L_(3,lane),cy[lane]); ax.text((R_(1,lane)+L_(3,lane))/2,cy[lane]+0.015,'no resampling',ha='center',va='bottom',fontsize=6,style='italic',color='#444')
A(ax,R_(3,2),cy[2],L_(4,2),cy[2])
# frozen objects flow downwards
for c,lab_ in ((1,'frozen $S$, features'),(3,'$f_m$')):
    Ad(ax,mid(c),cy[0]-H/2,mid(c),cy[1]+H/2); Ad(ax,mid(c),cy[1]-H/2,mid(c),cy[2]+H/2)
ax.text(mid(1)+0.01,(cy[0]+cy[1])/2,'$S$, features',fontsize=6.2,va='center'); ax.text(mid(1)+0.01,(cy[1]+cy[2])/2,'$S$, features',fontsize=6.2,va='center')
ax.text(mid(3)+0.01,(cy[0]+cy[1])/2,'$f_m$',fontsize=6.2,va='center'); ax.text(mid(3)+0.01,(cy[1]+cy[2])/2,'$f_m$, $\\theta^*$, $w^*$',fontsize=6.2,va='center')
ax.text(0.5,0.03,'dashed arrows: objects fitted earlier and passed on unchanged.\nNSL-KDD, UNSW-NB15: official training file split 85 / 15 (stratified); official test file as $D_{test}$.',ha='center',va='center',fontsize=5.9,style='italic',color='#333')
f.savefig('figs/fig_method.png'); plt.close(f)
# ======== result figures: shared house style ========
from matplotlib.lines import Line2D
NM={'RF':'Random Forest','SVM':'SVM (RBF)','XGB':'XGBoost','LSTM':'LSTM Network','kNN':'k-NN','Ensemble':'Proposed Ensemble'}
ST={'Ensemble':dict(color='#1f4fd8',ls='-',marker='D',lw=2.1,ms=4.2,z=10),
    'XGB':dict(color='#e08a00',ls='--',marker='s',lw=1.4,ms=4,z=6),
    'LSTM':dict(color='#7b3fe4',ls='-.',marker='^',lw=1.4,ms=4.3,z=5),
    'RF':dict(color='#0f9d73',ls=':',marker='o',lw=1.7,ms=4,z=4),
    'SVM':dict(color='#d6336c',ls=(0,(5,1.5,1,1.5)),marker='v',lw=1.3,ms=4,z=3),
    'kNN':dict(color='#6c757d',ls=(0,(1,1.6)),marker='X',lw=1.5,ms=4,z=2)}
ORDER=['Ensemble','XGB','RF','LSTM','SVM','kNN']
def style(a,grid=True):
    a.set_facecolor('#fbfbfd')
    for s in ('top','right'): a.spines[s].set_visible(True); a.spines[s].set_color('#c9ced6'); a.spines[s].set_linewidth(0.6)
    for s in ('left','bottom'): a.spines[s].set_linewidth(1.2); a.spines[s].set_color('#222')
    if grid: a.grid(True,ls=(0,(4,3)),lw=0.6,color='#d5dbe3'); a.set_axisbelow(True)
    a.tick_params(labelsize=7.5,width=0.9,length=3)
def lab(a,x=None,y=None,fs=8.5):
    if x: a.set_xlabel(x,fontsize=fs,fontweight='normal',labelpad=4)
    if y: a.set_ylabel(y,fontsize=fs,fontweight='normal',labelpad=4)
def leg(a,handles,title,loc,fs=7.5,**kw):
    lg=a.legend(handles=handles,title=title,loc=loc,fontsize=fs,title_fontsize=fs+1,frameon=True,fancybox=False,edgecolor='#9db4d6',framealpha=0.97,borderpad=0.6,labelspacing=0.45,handlelength=2.6,**kw)
    lg.get_title().set_fontweight('normal'); lg.get_frame().set_linewidth(0.9)
    for t in lg.get_texts():
        if t.get_text().startswith('Proposed'): t.set_fontweight('bold'); t.set_color('#1f4fd8')
    return lg
def H(m,txt=None,marker=True): 
    s=ST[m]; return Line2D([],[],color=s['color'],ls=s['ls'],lw=s['lw'],marker=s['marker'] if marker else None,ms=s['ms'],label=txt or NM[m])
SZ.update({'fig5':(6.4,2.8),'fig6':(5.2,2.8),'fig7':(5.3,3.9),'fig8':(6.4,3.4),'fig9':(5.2,3.7),'fig10':(5.2,2.9),'fig11':(5.8,3.3)})
with plt.rc_context({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','Arial','DejaVu Sans']}):
    # ---- Fig 5 class-conditional behaviour (LEACH-Sim, CH rows)
    df=pd.read_csv('leach_sim.csv' if os.path.exists('leach_sim.csv') else 'leach_ch_rows.csv.gz'); ch=df[df.is_ch==1].copy(); ch['fwd']=np.where(ch.data_recv>0,ch.bs_recv/ch.data_recv.clip(lower=1),np.nan)
    order=['Normal','Blackhole','Grayhole','Flooding','Scheduling']; short=['Normal','Black-\nhole','Gray-\nhole','Flood-\ning','Sched-\nuling']
    f,axs=plt.subplots(1,3,figsize=SZ['fig5'])
    for a,(col,t) in zip(axs,[('fwd','BS-received / CH-received data'),('adv_sent','ADV messages sent'),('data_recv','Data packets received by CH')]):
        dat=[ch[ch.label_name==c][col].dropna().values for c in order]
        bp=a.boxplot(dat,showfliers=False,patch_artist=True,widths=0.62,medianprops=dict(color='#111',lw=1.4),boxprops=dict(lw=0.9),whiskerprops=dict(lw=0.9),capprops=dict(lw=0.9))
        for p_,c_ in zip(bp['boxes'],['#8fd3b6','#f08a8a','#f6b26b','#8fb8f0','#b9a7ee']): p_.set_facecolor(c_)
        a.set_xticklabels(short,fontsize=6.2); a.set_title(t,fontsize=8,fontweight='normal'); style(a)
    f.tight_layout(); f.savefig('figs/fig5.png'); plt.close(f)
    # ---- Fig 6 learning curves
    h=L['lstm_hist']; ep=np.arange(1,len(h['val_loss'])+1); f,a=plt.subplots(figsize=SZ['fig6']); a2=a.twinx()
    a.plot(ep,np.array(h['train_acc'])*100,color='#1f4fd8',lw=2.2,marker='D',ms=3.6,label='Training accuracy'); a.plot(ep,np.array(h['val_acc'])*100,color='#1f4fd8',ls='--',lw=1.5,marker='s',ms=3.4,label='Validation accuracy')
    a2.plot(ep,h['train_loss'],color='#d6336c',lw=1.6,ls='-.',marker='^',ms=3.4,label='Training loss'); a2.plot(ep,h['val_loss'],color='#e08a00',lw=1.5,ls=':',marker='o',ms=3.4,label='Validation loss')
    a.margins(y=0.08); a2.margins(y=0.08); a.axvline(L['lstm_best_epoch'],color='#222',ls=(0,(2,2)),lw=0.9); style(a); a2.tick_params(labelsize=7.5); a2.spines['right'].set_visible(True); a2.spines['right'].set_linewidth(1.2); a2.spines['right'].set_color('#222')
    a.text(L['lstm_best_epoch']-0.3,np.mean(a.get_ylim()),f"best epoch {L['lstm_best_epoch']}",fontsize=7.2,rotation=90,ha='right',va='center')
    lab(a,'Epoch','Accuracy (%)'); a2.set_ylabel('Cross-entropy loss',fontsize=8.5)
    hs=a.get_legend_handles_labels(); h2=a2.get_legend_handles_labels(); leg(a,hs[0]+h2[0],'LSTM training','center',fs=7,bbox_to_anchor=(0.42,0.5))
    f.tight_layout(); f.savefig('figs/fig6.png'); plt.close(f)
    # ---- Fig 7 ROC attack-vs-normal (LEACH)
    y=_yte('leach'); yb=(y!=0).astype(int); f,a=plt.subplots(figsize=SZ['fig7']); ins=a.inset_axes([0.47,0.45,0.50,0.38]); AU_={}
    for m in ORDER[::-1]:
        s=_score('leach',m,0); fpr,tpr,_=roc_curve(yb,s); AU_[m]=auc(fpr,tpr); st=ST[m]
        a.plot(fpr,tpr,color=st['color'],ls=st['ls'],lw=st['lw'],zorder=st['z']); ins.plot(fpr,tpr,color=st['color'],ls=st['ls'],lw=st['lw'],zorder=st['z'])
    a.plot([0,1],[0,1],ls=(0,(5,4)),color='#5b7fb5',lw=1); a.set_xlim(-0.025,1.0); a.set_ylim(-0.025,1.045); a.set_xticks([0,0.2,0.4,0.6,0.8,1.0]); a.set_yticks([0,0.2,0.4,0.6,0.8,1.0]); lab(a,'False Positive Rate (FPR)','True Positive Rate (TPR)'); style(a)
    ins.set_xlim(-0.0006,0.02); ins.set_ylim(0.90,1.008); ins.set_xticks([0,0.01,0.02]); ins.set_yticks([0.90,0.95,1.0]); ins.tick_params(labelsize=6.5); ins.grid(True,ls=(0,(4,3)),lw=0.5,color='#d5dbe3')
    ins.set_title('Zoom: low-FPR, high-TPR region',fontsize=7.5,pad=3)
    for s_ in ins.spines.values(): s_.set_linewidth(1.0); s_.set_color('#222')
    a.indicate_inset_zoom(ins,edgecolor='#222',linestyle=':',alpha=0.9)
    leg(a,[H(m,f'{NM[m]}: {AU_[m]:.5f}',marker=False) for m in ORDER],'Model AUC','lower left',fs=7,bbox_to_anchor=(0.03,0.04))
    f.tight_layout(); f.savefig('figs/fig7.png'); plt.close(f)
    # ---- Fig 8 two panels
    f,(a1,a2)=plt.subplots(1,2,figsize=SZ['fig8'])
    ns=L['noise_sweep']; xk=sorted(int(k) for k in ns)
    for m in ORDER[::-1]:
        st=ST[m]; xs,ys=zip(*L['malfrac_sweep'][m]); kw=dict(color=st['color'],ls=st['ls'],lw=st['lw'],marker=st['marker'],ms=st['ms'],zorder=st['z'])
        a1.plot(xs,np.array(ys)*100,**kw); a2.plot(xk,[ns[str(x)][m]*100 for x in xk],**kw)
    for a in (a1,a2): style(a)
    lab(a1,'Malicious nodes in the test network (%)','Macro-F1 (%)',fs=8); lab(a2,'Extra packet loss in test features (%)','Macro-F1 (%)',fs=8)
    a1.set_title('(a) Share of malicious nodes',fontsize=8.5); a2.set_title('(b) Synthetic packet loss',fontsize=8.5)
    lg=f.legend(handles=[H(m) for m in ORDER],title='Model comparison',loc='lower center',ncol=6,fontsize=6.8,title_fontsize=7.5,frameon=True,fancybox=False,edgecolor='#9db4d6',handlelength=2.2,columnspacing=1.1,borderpad=0.5)
    [t.set_fontweight('bold') or t.set_color('#1f4fd8') for t in lg.get_texts() if t.get_text().startswith('Proposed')]
    f.tight_layout(rect=(0,0.15,1,1)); f.savefig('figs/fig8.png'); plt.close(f)
    # ---- Fig 9 PR curves NSL-KDD attack-vs-normal
    y=_yte('nsl'); nid=N['classes'].index('Normal'); yb=(y!=nid).astype(int); f,a=plt.subplots(figsize=SZ['fig9']); AP_={}
    xx=np.linspace(0.31,1,400)
    for fv in (0.6,0.7,0.8,0.9):
        yy=fv*xx/(2*xx-fv); ok=(yy>0)&(yy<=1.0)&(xx>fv/2+1e-3); a.plot(xx[ok],yy[ok],color='#9fb1c7',lw=0.8,ls=(0,(1,2)),zorder=1)
        a.text(1.012,fv/(2-fv),f'F1={fv}',fontsize=6.3,color='#5f7391',ha='left',va='center',clip_on=False)
    for m in ORDER[::-1]:
        s=_score('nsl',m,nid); p_,r_,_=precision_recall_curve(yb,s); AP_[m]=average_precision_score(yb,s); st=ST[m]
        a.plot(r_,p_,color=st['color'],ls=st['ls'],lw=st['lw'],zorder=st['z'])
    a.set_xlim(0,1.0); a.set_ylim(0.4,1.02); a.set_xticks([0,0.2,0.4,0.6,0.8,1.0]); lab(a,'Recall (Detection Sensitivity)','Precision'); style(a)
    leg(a,[H(m,f'{NM[m]}: {AP_[m]:.3f}',marker=False) for m in ORDER],'Average Precision (AP)','lower left',fs=7,bbox_to_anchor=(0.03,0.04))
    f.tight_layout(); f.savefig('figs/fig9.png'); plt.close(f)
    # ---- Fig 10 UNSW-NB15 confusion matrix (row-normalised) of the ensemble
    from sklearn.metrics import confusion_matrix as _cm, f1_score as _f1
    from matplotlib.colors import LinearSegmentedColormap
    yu=_yte('unsw'); cls=U['classes']; K=len(cls); pr=_unsw_pred()
    cmx=_cm(yu,pr,labels=range(K)); sup=cmx.sum(1); cn=cmx/sup[:,None]*100; f1c=_f1(yu,pr,average=None,labels=range(K),zero_division=0)
    SZ['fig10']=(6.2,4.9); f,a=plt.subplots(figsize=SZ['fig10'])
    cmap=LinearSegmentedColormap.from_list('b',['#ffffff','#cfdcf7','#6d94ea','#1f4fd8','#0b2a8a'])
    im=a.imshow(cn,cmap=cmap,vmin=0,vmax=100,aspect='equal')
    for i in range(K):
        for j in range(K):
            v=cn[i,j]
            if cmx[i,j]==0: continue
            a.text(j,i,f'{v:.0f}' if v>=1 or v==0 else '<1',ha='center',va='center',fontsize=6.6,color='white' if v>55 else '#222')
    a.set_xticks(range(K)); a.set_yticks(range(K)); a.set_xticklabels(cls,rotation=40,ha='right',fontsize=7.2); a.set_yticklabels([f'{c} (n={n:,})' for c,n in zip(cls,sup)],fontsize=7)
    a.set_xticks(np.arange(-.5,K,1),minor=True); a.set_yticks(np.arange(-.5,K,1),minor=True); a.grid(which='minor',color='#e3e8f0',lw=0.6); a.grid(which='major',visible=False); a.tick_params(which='minor',length=0)
    for sp in a.spines.values(): sp.set_visible(True); sp.set_linewidth(0.9); sp.set_color('#222')
    a.set_xlabel('Predicted class',fontsize=8.5,labelpad=4); a.set_ylabel('True class',fontsize=8.5,labelpad=4)
    a.text(K-0.35,-0.95,'F1',fontsize=7.2,ha='left',va='center',clip_on=False)
    for i in range(K): a.text(K-0.35,i,f'{f1c[i]:.2f}',fontsize=7.2,ha='left',va='center',clip_on=False)
    cb=f.colorbar(im,ax=a,fraction=0.035,pad=0.14,shrink=0.85); cb.set_label('Share of true class predicted as column (%)',fontsize=7.5); cb.ax.tick_params(labelsize=7)
    f.tight_layout(); f.savefig('figs/fig10.png',bbox_inches='tight',pad_inches=0.06); plt.close(f)
    # ---- Fig 11 latency vs throughput
    f,a=plt.subplots(figsize=SZ['fig11'])
    for m in ORDER[::-1]:
        c=L['cost'][m]; st=ST[m]; big=m=='Ensemble'
        a.scatter(c['throughput_sps'],c['latency_ms'],s=95 if big else 48,color=st['color'],marker=st['marker'],edgecolor='#111',linewidth=0.8,zorder=st['z']+5)
        a.annotate(NM[m],(c['throughput_sps'],c['latency_ms']),textcoords='offset points',xytext=(7,5),fontsize=8 if big else 7.5,fontweight='bold' if big else 'normal',color='#1f4fd8' if big else '#222')
    a.set_xscale('log'); a.set_yscale('log'); lab(a,'Batch throughput (samples per second, log scale)','Single-sample latency (ms, log scale)'); style(a)
    a.text(0.98,0.03,'Measured on the x86 build CPU,\nnot on sensor hardware',transform=a.transAxes,fontsize=7,style='italic',ha='right',va='bottom')
    f.tight_layout(); f.savefig('figs/fig11.png'); plt.close(f)
print('ok')
