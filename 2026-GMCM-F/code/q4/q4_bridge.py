"""Separated Loss scales and explicit baselines, with out-of-fold predictions."""
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from data_config import BRIDGE_SCORES
from q4_data import save_table


MODELS=['training_mean','logN_linear','loss_linear','loss_ridge10']


def fit_predict(train,test,model):
    y=train.LB_Average.to_numpy(float);n=len(train)
    if model=='training_mean':
        x=np.ones((n,1));xt=np.ones((len(test),1));penalty=0.
    else:
        a,b=(np.log10(train.N_params_B),np.log10(test.N_params_B)) if model=='logN_linear' else (train.Val_Loss,test.Val_Loss)
        center=float(np.mean(a));scale=float(np.std(a))
        if scale<=0:raise ValueError('Constant bridge covariate')
        x=np.column_stack([np.ones(n),(a-center)/scale]);xt=np.column_stack([np.ones(len(test)),(b-center)/scale])
        penalty=10. if model=='loss_ridge10' else 0.
    reg=np.diag([0.]+[penalty]*(x.shape[1]-1));inv=np.linalg.inv(x.T@x+reg)
    coef=inv@x.T@y;pred=xt@coef;res=y-x@coef
    effective_df=float(np.trace(x@inv@x.T));dof=n-effective_df
    variance=float(res@res/dof)
    leverage=np.sum((xt@inv)*(xt@inv@(x.T@x)),axis=1)
    half=student_t.ppf(.95,dof)*np.sqrt(variance*(1+leverage))
    return pred,pred-half,pred+half


def bridge_analysis(c6,b1):
    rows=[];press_checks=[];summary=[]
    for level in ['High','Medium']:
        d=c6[c6.Loss_Comparability.str.startswith(level)].copy().reset_index(drop=True)
        d['publisher']=d.Model.str.casefold().str.split('/').str[0]
        groups=np.arange(len(d)).astype(str) if level=='High' else d.publisher.to_numpy()
        for group in np.unique(groups):
            train=d[groups!=group];test=d[groups==group]
            for model in MODELS:
                pred,lo,hi=fit_predict(train,test,model)
                for i,r in enumerate(test.itertuples()):
                    rows.append({'comparability':level,'validation':'leave_one_model_out' if level=='High' else 'leave_publisher_out',
                                 'held_out_group':group,'model':model,'repository':r.Model,'publisher':r.publisher,
                                 'actual':r.LB_Average,'prediction':pred[i],'pi90_low':lo[i],'pi90_high':hi[i],
                                 'train_rows':len(train),'test_rows':len(test),'Val_Loss':r.Val_Loss,'N_B':r.N_params_B,
                                 'PI_assumption':'normal independent errors; ridge interval additionally ignores shrinkage bias'})
        if level=='High':
            # Independent PRESS check against explicit refits; ridge scales are refit within folds.
            for model in MODELS[:3]:
                z=np.log10(d.N_params_B) if model=='logN_linear' else d.Val_Loss
                x=np.ones((len(d),1)) if model=='training_mean' else np.column_stack([np.ones(len(d)),z])
                beta=np.linalg.lstsq(x,d.LB_Average,rcond=None)[0];hat=np.sum(x*(x@np.linalg.inv(x.T@x)),axis=1)
                loo=d.LB_Average-(d.LB_Average-x@beta)/(1-hat)
                got=pd.DataFrame(rows);got=got[got.comparability.eq('High')&got.model.eq(model)].set_index('repository').loc[d.Model,'prediction'].to_numpy()
                press_checks.append({'model':model,'maximum_PRESS_refit_difference':float(np.max(abs(loo-got)))})
    df=pd.DataFrame(rows);save_table(df,'bridge_out_of_fold_predictions.csv')
    for (level,model),g in df.groupby(['comparability','model']):
        e=g.prediction-g.actual
        summary.append({'comparability':level,'model':model,'validation':g.validation.iloc[0],'n':len(g),'held_out_groups':g.held_out_group.nunique(),
                        'RMSE_pp':float(np.sqrt(np.mean(e**2))),'MAE_pp':float(np.mean(abs(e))),'bias_pp':float(e.mean()),
                        'PI90_coverage':float(((g.actual>=g.pi90_low)&(g.actual<=g.pi90_high)).mean()),
                        'PI90_mean_width_pp':float((g.pi90_high-g.pi90_low).mean())})
    metrics=pd.DataFrame(summary);save_table(metrics,'bridge_metrics.csv');save_table(pd.DataFrame(press_checks),'bridge_PRESS_verification.csv')
    n,d=1.,150.
    loss=b1['E']+b1['A']*n**(-b1['alpha'])+b1['B']*d**(-b1['beta'])
    high=c6[c6.Loss_Comparability.str.startswith('High')].copy()
    test=pd.DataFrame({'N_params_B':[n],'Val_Loss':[loss]});examples=[]
    for model in MODELS:
        p,l,u=fit_predict(high,test,model)
        examples.append({'B1_N_B':n,'B1_D_B':d,'B1_Loss':loss,'model':model,'score':p[0],'PI90_low':l[0],'PI90_high':u[0],
                         'inside_high_loss_range':bool(high.Val_Loss.min()<=loss<=high.Val_Loss.max()),
                         'interpretation':'conditional illustration; high-comparability sample has 7 points in one family'})
    save_table(pd.DataFrame(examples),'B1_loss_bridge_example.csv')
    return {'high_rows':int(c6.Loss_Comparability.str.startswith('High').sum()),
            'six_score_average_max_abs_error':float(abs(c6[BRIDGE_SCORES].mean(axis=1)-c6.LB_Average).max()),
            'PRESS_refit_max_difference':max(r['maximum_PRESS_refit_difference'] for r in press_checks),
            'metrics':summary,'B1_reference_Loss':float(loss),'B6_Loss_transferred':False,
            'limits':'High is one 7-model family; Medium has incompatible approximate Loss scales; PI normal-error assumptions are not guaranteed'},metrics
