from os import cpu_count
from re import M, S
from matplotlib.pyplot import bar
from scipy.optimize import curve_fit
from jetset.analytical_model import AnalyticalParameter
from jetset.model_parameters import ModelParameterArray
from jetset.minimizer import LSMinimizer, ModelMinimizer,fit_XY
from jetset.data_loader import Data
from jetset.model_manager import FitModel
import pylab as plt
from jetset.base_model import Model
import numpy as np
import pickle
from astropy.table import Table, vstack
from itertools import cycle
import copy

class DecayLC(Model):
    """
    Class to handle function for growth model
    """
    
    def __init__(self,nu_size=100,**keywords):
        """
        """
     
        super(DecayLC,self).__init__(  **keywords)
        self.name='Exponential'
        self.nu_min=-100
        self.nu_max=100
        self.parameters = ModelParameterArray()      
        self.F0=1
        self.parameters.add_par(AnalyticalParameter(self,name='F0',par_type='',val=1,val_min=0,val_max=None,units=''))        
        self.tau_d=1
        self.parameters.add_par(AnalyticalParameter(self,name='tau_d',par_type='',val=1,val_min=0.,val_max=None,units=''))
        self.t_0=1
        self.parameters.add_par(AnalyticalParameter(self,name='t_0',par_type='',val=1,val_min=0,val_max=None,units=''))
    
    def set(self,**keywords):
        #super(GrowthMode,self).set(**keywords )

        """
        sets a parameter value checking for physical boundaries 
        """
        if 'val' in keywords.keys():
            self.assign_val(self.name,keywords['val']) 

    def assign_val(self,name,val):
        setattr(self.polymodel,name,val)
    
    def lin_func(self,nu):
        return self.F0*np.exp(-(nu-self.t_0)/self.tau_d)


def do_decay_fit(lc,name='',minimizer='minuit',t_start_frac=0.8, t_stop_frac=0.5,show_report=True):
    
    t=lc['time']
    y=lc['flux']
    

    data=Data(n_rows=len(lc))
    data.set_field('x',t)
    data.set_field('y',y)
    data.set_field('dy',y*0.1)
    

    F0=y.max()
    tmax=t[np.argmax(y)]
    print('t_max=%2.2f'%tmax)
    id_start = np.argwhere(y>F0*t_start_frac).ravel().max()
    id_stop = np.argwhere(y>F0*t_stop_frac).ravel().max()
    
    t_start=t[id_start]
    t_stop=t[id_stop]
    t_0=t_start
    exp_model=DecayLC()

    fm=FitModel(analytical=exp_model,name='test')
    fm.nu_min_fit=t_start
    fm.nu_max_fit=t_stop
    fm.parameters
    fm.components.Exponential.parameters.F0.val=F0
    fm.components.Exponential.parameters.t_0.val=t_0
    fm.components.Exponential.parameters.tau_d.val=1
    fm.components.Exponential.parameters.t_0.fit_range=[max(0,t_0/10),t_0*10]
    fm.components.Exponential.parameters.F0.fit_range=[F0/100,F0*100]

    bfm,mm=fit_XY(fm,data,x_fit_start=t_start,x_fit_stop=t_stop,minimizer=minimizer,silent=True)
    if show_report is True:
        mm.show_report()
    
    fig=plt.figure(figsize=(10,5),dpi=100)

    plt.plot(t,y)
    _t=np.linspace(t_start,t_stop,100)
    plt.plot(_t,fm.eval(nu=_t,get_model=True), label='tau =%2.2f %s'%(fm.components.Exponential.parameters.tau_d.val,name))
    plt.xlabel('t')
    plt.ylabel('f')
    plt.legend()

    
    return tmax,fm.components.Exponential.parameters.tau_d.val,F0




class RespConvolveLC(Model):
    """
    Class to handle function for growth model
    """
    
    def __init__(self,nu_size=100,**keywords):
        """
        """
     
        super(RespConvolveLC,self).__init__(  **keywords)
        self.name='Exponential'
        self.nu_min=-100
        self.nu_max=100
        self.parameters = ModelParameterArray()      
        self.parameters.add_par(AnalyticalParameter(self,name='delta_T',par_type='',val=1,val_min=0,val_max=None,units='d'))        
        self.parameters.add_par(AnalyticalParameter(self,name='t_decay',par_type='',val=1,val_min=0.,val_max=None,units='d'))
        self.parameters.add_par(AnalyticalParameter(self,name='phi',par_type='',val=1,val_min=0.,val_max=None,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='t_rise',par_type='',val=1,val_min=0.,val_max=None,units='d'))
        self.parameters.add_par(AnalyticalParameter(self,name='psi',par_type='',val=1,val_min=0.,val_max=None,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='A',par_type='',val=1,val_min=0,val_max=None,units=''))
        
        self.lc_in=None
    
    def _S(self,t,delta_T,t_decay,phi,t_rise,psi,A):
        a1=1/(1+np.exp(-(t-delta_T)/t_rise))
        a2=np.exp(-(t-(delta_T))/t_decay)
        return A*(a1**psi)*(a2**phi)

    def _S_1(self,t,delta_T,t_decay,phi,t_rise,psi,A):
        a1=np.exp((delta_T-t)/t_rise)
        a2=np.exp((t-delta_T)/t_decay)
        return A/(a1+a2)
       
    
    def lin_func(self,nu):
        s=self._S(self.lc_in['time'],
                  self.parameters.delta_T.val,
                  self.parameters.t_decay.val,
                  self.parameters.phi.val,
                  self.parameters.t_rise.val,
                  self.parameters.psi.val,
                  self.parameters.A.val)
        flux_array_out=np.convolve(self.lc_in['flux'],s)
        dt=self.lc_in['time'][1]-self.lc_in['time'][0]
        time_array_out=np.arange(1,flux_array_out.size+1)*dt

        return np.interp(nu,xp=time_array_out,fp=flux_array_out)


def merge_lc(lc_falre,lc_exp):
    
    lc_falre=copy.deepcopy(lc_falre)
    lc_exp=copy.deepcopy(lc_exp)
    delta_t=lc_falre['time'][1]-lc_falre['time'][0]

    lc_exp['time']=lc_exp['time']+lc_falre['time'][-1]+delta_t
    
    merged_lc=vstack([lc_falre,lc_exp])
              
    return merged_lc

def gamma_radio_delay_fit(infile,lc_name_1,lc_name_2,t_start_frac=0.8, t_stop_frac=0.5,flare_lc=None,phi_frozen=True,psi_frozen=True,max_points=None,t_exp=None,delta=None,R0=None,beta_exp=None,flare_duration=None):
    with open(infile, 'rb') as f:
        lcs_v_exp=pickle.load(f)
    
    
    if 'beta_exp' in lcs_v_exp.keys() and beta_exp is None:
        lcs_v_exp['beta_exp']=lcs_v_exp['beta_exp']

        beta_exp=lcs_v_exp['beta_exp']
        
        print('beta_exp',beta_exp)
    print('-'*40)
    lc_2=lcs_v_exp[lc_name_2]
    lc_1=lcs_v_exp[lc_name_1]
    
    
    if flare_lc is not None:
        with open(flare_lc, 'rb') as f:
            lcs_flare=pickle.load(f)
            
            lc_1_flare=lcs_flare[lc_name_1]
            lc_2_flare=lcs_flare[lc_name_2]

    
        lc_1=merge_lc(lc_1_flare,lc_1)
        lc_2=merge_lc(lc_2_flare,lc_2)


    lc_1['flux']=lc_1['flux']/lc_1['flux'].max()
    lc_1['time']=lc_1['time'].to('d')

    lc_2['flux']=lc_2['flux']/lc_2['flux'].max()
    lc_2['time']=lc_2['time'].to('d')

    t_max_g,t_d_g,Fm_g=do_decay_fit(lc_1,name=lc_1.meta['name'],minimizer='minuit',t_start_frac=0.75,t_stop_frac=0.1,show_report=False)
    t_max_r,t_d_r,Fm_r=do_decay_fit(lc_2,name=lc_2.meta['name'],minimizer='minuit',t_start_frac=0.8,t_stop_frac=0.2,show_report=False)

    delta_t=t_max_r-t_max_g
    t_decay=t_d_r
    
    A=Fm_r/Fm_g
    print('Fm_r/Fm_g',A)
    f_res=RespConvolveLC()
    f_res.parameters.phi.val=1
    f_res.parameters.phi.frozen=phi_frozen

    f_res.parameters.psi.val=1
    f_res.parameters.psi.frozen=psi_frozen    

    f_res.parameters.delta_T.val=delta_t
    f_res.parameters.delta_T.fit_range=[delta_t/10,delta_t*10]   
 
    f_res.parameters.t_decay.val=t_decay
    f_res.parameters.t_decay.fit_range=[t_decay*0.1,t_decay*2]   

    f_res.parameters.t_rise.val=t_decay/10
    f_res.parameters.t_rise.fit_range=[t_decay*0.01,t_decay*2]   

    f_res.parameters.A.val=1
    f_res.parameters.A.fit_range=[A/1000,A*1000]   

    f_res.parameters.A.frozen=False

    if len(lc_1)>max_points:
        lc_1_fit=lc_1.copy()
        lc_2_fit=lc_2.copy()
        time_array=np.interp(np.linspace(lc_1_fit['time'][0],lc_1_fit['time'][-1],max_points), lc_1_fit['time'], lc_1_fit['time'])
        l1=np.interp(time_array, lc_1_fit['time'], lc_1_fit['flux'],left=0, right=0)
        l2=np.interp(time_array, lc_2_fit['time'], lc_2_fit['flux'],left=0, right=0)
        lc_1_fit.remove_rows(slice(max_points, len(lc_1_fit)))
        lc_2_fit.remove_rows(slice(max_points, len(lc_2_fit)))

        lc_1_fit['time']=time_array
        lc_2_fit['time']=time_array
        lc_1_fit['flux']=l1
        lc_2_fit['flux']=l2
    else:
        lc_1_fit=lc_1
        lc_2_fit=lc_2

    f_res.lc_in=lc_1_fit

    fm=FitModel(analytical=f_res,name='test')
    data=Data(n_rows=len(lc_2_fit))
    data.set_field('x',lc_2_fit['time'])
    data.set_field('y',lc_2_fit['flux'])
    data.set_field('dy',lc_2_fit['flux']*0.1)

    F0=lc_2['flux'].max()
    id_start = np.argwhere(lc_2['flux']>F0*t_start_frac).ravel().min()
    id_stop = np.argwhere(lc_2['flux']>F0*t_stop_frac).ravel().max()
    
    t_start=lc_2['time'][id_start]
    t_stop=lc_2['time'][id_stop]

    if t_exp is not None and delta is not None and flare_duration is not None:
        x_fit_start=max(t_exp/86400/delta+1.5*flare_duration/86400/delta,t_start)
    else:
        x_fit_start=t_start
    x_fit_stop=t_stop
    

    
    bfm,mm=fit_XY(fm,data,x_fit_start=x_fit_start,x_fit_stop=x_fit_stop,minimizer='minuit',silent=False)

    fig=plt.figure(dpi=120)

    t=lc_2['time']

    plt.plot(t,bfm.fit_model.eval(nu=t,get_model=True),label='$l_R(t)=S(t)*l_{\gamma}(t)$')
    t=t[t>x_fit_start]
    t=t[t<x_fit_stop]

    plt.plot(lc_1['time'],lc_1['flux'],label=lc_1.meta['name'].replace('_',' '))
    plt.plot(lc_2['time'],lc_2['flux'],label=lc_2.meta['name'].replace('_',' '))
    plt.plot(t,bfm.fit_model.eval(nu=t,get_model=True),label='fit range',ls='--')

    plt.xlabel(r'$t^{\rm obs}$  (d)')
    plt.ylabel('F/Fmax ')
    plt.legend(loc='best')
    plt.title('$\phi=%2.1f, \Delta_t=%2.1f, t_{decay}=%2.1f$'%(f_res.parameters.phi.val,f_res.parameters.delta_T.val,f_res.parameters.t_decay.val))
    plt.show()

    tm1= (f_res.parameters.t_decay.val+f_res.parameters.t_rise.val)/(f_res.parameters.t_decay.val-f_res.parameters.t_rise.val)
    tm=f_res.parameters.delta_T.val+tm1
    print('delta_T,tm',f_res.parameters.delta_T.val,t_max_r)
    if t_exp is not None and delta is not None and R0 is not None and beta_exp is not None:
        t=t_max_r*delta*86400-t_exp
        R_t=t*beta_exp*3E10+R0
        print('R at peak=%e'%R_t)
    print('-'*40)
    
    return beta_exp,\
           delta_t,\
           f_res.parameters.delta_T.val,\
           f_res.parameters.delta_T.best_fit_err,\
           t_decay,\
           f_res.parameters.t_decay.val,\
           f_res.parameters.t_decay.best_fit_err,\
           f_res.parameters.t_rise.val,\
           f_res.parameters.t_rise.best_fit_err,\
           fig

 


def func_t_dec(x, R0):
    return  R0/(x*3E10)

def func_t_rise(x,Rtr):
    return Rtr/(x*3E10)

def func_delta_T(x, t_exp, R0):
    return t_exp +(R0/(x*3E10))

def eval_p(psi,m_B):
    return -2*((m_B + 2)*psi - 2)/(m_B*psi - 1)

class ExpDeltaT_vexp(Model):
    """
    Class to handle function for growth model
    """
    
    def __init__(self,nu_size=100,**keywords):
        """
        """
     
        super(ExpDeltaT_vexp,self).__init__(  **keywords)
        self.name='DeltaT'
        self.nu_min=-100
        self.nu_max=100
        self.parameters = ModelParameterArray()      
        self.parameters.add_par(AnalyticalParameter(self,name='t_exp',par_type='',val=1,val_min=0.,val_max=None,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='R0',par_type='',val=1,val_min=0,val_max=None,units='cm'))
        self.parameters.add_par(AnalyticalParameter(self,name='nu_0',par_type='',val=1E10,val_min=1E9,val_max=1E12,units='Hz'))
        self.parameters.add_par(AnalyticalParameter(self,name='nu_1',par_type='',val=1E10,val_min=1E6,val_max=1E15,units='Hz'))
        self.parameters.add_par(AnalyticalParameter(self,name='phi',par_type='',val=0.5,val_min=-2,val_max=2,units=''))
        
    
    def lin_func(self,nu):
        a=(1)*(self.parameters.R0.val/(3E10*nu))       
        phi=self.parameters.phi.val
        a=a*((self.parameters.nu_0.val/self.parameters.nu_1.val)**(phi)-1)
        a[a<=0]=0
        return self.parameters.t_exp.val + a
    
class ExpTrise(Model):
    """
    Class to handle function for growth model
    """
    
    def __init__(self,nu_size=100,**keywords):
        """
        """
     
        super(ExpTrise,self).__init__(  **keywords)
        self.name='ExpTrise'
        self.nu_min=-100
        self.nu_max=100
        self.parameters = ModelParameterArray()      
        self.parameters.add_par(AnalyticalParameter(self,name='R0',par_type='',val=1,val_min=0,val_max=None,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='nu_0',par_type='',val=1E10,val_min=1E9,val_max=1E12,units='Hz'))
        self.parameters.add_par(AnalyticalParameter(self,name='nu_1',par_type='',val=1E10,val_min=1E6,val_max=1E15,units='Hz'))
        self.parameters.add_par(AnalyticalParameter(self,name='phi',par_type='',val=0.5,val_min=0,val_max=1,units=''))
    
    def lin_func(self,nu):
        a=(1/3)*(self.parameters.R0.val/(3E10*nu))
        phi=self.parameters.phi.val
        a=a*((self.parameters.nu_0.val/self.parameters.nu_1.val)**(phi)-1)

        a[a<=0]=0
        return a

    
    
class ExpTdec_vexp(Model):
    """
    Class to handle function for growth model
    """
    
    def __init__(self,nu_size=100,**keywords):
        """
        """
        super(ExpTdec_vexp,self).__init__(  **keywords)
        self.nu_min=-100
        self.nu_max=100
        self.name='ExpTdec_vexp'
        self.parameters = ModelParameterArray()      
        
        self.parameters.add_par(AnalyticalParameter(self,name='R0',par_type='',val=1,val_min=0,val_max=None,units='cm'))        
        self.parameters.add_par(AnalyticalParameter(self,name='m_B',par_type='',val=1.0,val_min=0.5,val_max=2.5,units=''))

        self.parameters.add_par(AnalyticalParameter(self,name='nu_0',par_type='',val=1E10,val_min=1E9,val_max=1E12,units='Hz'))
        self.parameters.add_par(AnalyticalParameter(self,name='nu_1',par_type='',val=1E10,val_min=1E6,val_max=1E15,units='Hz'))
        self.parameters.add_par(AnalyticalParameter(self,name='phi',par_type='',val=0.5,val_min=0,val_max=1,units=''))
        
   
    def lin_func(self,nu):
       
        a=(self.parameters.R0.val/(3E10*nu))
        a=a/(2*self.parameters.m_B.val)
        phi=self.parameters.phi.val
        a=a*(self.parameters.nu_0.val/self.parameters.nu_1.val)**(phi)
        return a
        

class ExpTdec_vexp_beta_S(Model):
    """
    Class to handle function for growth model
    """
    
    def __init__(self,nu_size=100,**keywords):
        """
        """
        super(ExpTdec_vexp_beta_S,self).__init__(  **keywords)
        self.nu_min=-100
        self.nu_max=100
        self.name='ExpTdec_vexp_beta_S'
        self.parameters = ModelParameterArray()      
        
        self.parameters.add_par(AnalyticalParameter(self,name='R0',par_type='',val=1,val_min=0,val_max=None,units='cm'))        
        self.parameters.add_par(AnalyticalParameter(self,name='m_B',par_type='',val=1.0,val_min=0.5,val_max=2.5,units=''))

        self.parameters.add_par(AnalyticalParameter(self,name='nu_0',par_type='',val=1E10,val_min=1E9,val_max=1E12,units='Hz'))
        self.parameters.add_par(AnalyticalParameter(self,name='nu_1',par_type='',val=1E10,val_min=1E6,val_max=1E15,units='Hz'))
        self.parameters.add_par(AnalyticalParameter(self,name='phi',par_type='',val=0.5,val_min=0.0,val_max=1.9,units=''))

        self.parameters.add_par(AnalyticalParameter(self,name='beta_S',par_type='',val=0.01,val_min=0.001,val_max=0.1,units=''))
        
   
    def lin_func(self,nu):
        
        a=(self.parameters.R0.val/(3E10*nu))
        a=a/(2*self.parameters.m_B.val)
        phi=self.parameters.phi.val
        a=a*(self.parameters.nu_0.val/self.parameters.nu_1.val)**(phi)
        a=a*(1/(1+np.exp(-(nu-self.parameters.beta_S.val)/self.parameters.beta_S.val)))
        return a
        


def gamma_radio_delay_analysis_vs_v_exp(beta_exp,
                                        t_decay_days,
                                        t_decay_days_err,
                                        t_rise_days,
                                        t_rise_days_err,
                                        delta_T_days,
                                        delta_T_days_err,
                                        t_decay_gamma_days,
                                        R0_cm,
                                        t_exp_sym,
                                        delta):

    x=np.linspace(beta_exp.min()*0.9,min(0.9,beta_exp.max()*1.1),100)
    t_decay_days=t_decay_days+t_decay_gamma_days
    

    print('------ ExpTrise')
    data=Data(n_rows=beta_exp.size)
    data.set_field('x' ,beta_exp)
    data.set_field('y' ,t_rise_days*86400)
    data.set_field('dy',t_rise_days_err*86400)

    fit_func=ExpTrise()
    nu0=2E11
    fm=FitModel(analytical=fit_func,name='test')
    fm.nu_min_fit=beta_exp.min()
    fm.nu_max_fit=beta_exp.max()
    R_tr_input=R0_cm/delta
    fm.components.ExpTrise.parameters.R0.val=R_tr_input
    fm.components.ExpTrise.parameters.R0.fit_range=[R_tr_input/10,R_tr_input*10]
    fm.components.ExpTrise.parameters.nu_1.val=1.5E10
    fm.components.ExpTrise.parameters.nu_1.frozen=True
    fm.components.ExpTrise.parameters.nu_0.val=nu0
    fm.components.ExpTrise.parameters.nu_0.fit_range=[1E9,1E12]
    beta_min=beta_exp.min()

    bfm,mm=fit_XY(fm,data,x_fit_start=beta_min,x_fit_stop=beta_exp.max(),minimizer='minuit',silent=True)
    mm.show_report()
    
    f_trise=plt.figure(dpi=100)
    plt.errorbar( beta_exp , t_rise_days,yerr=t_rise_days_err, ls='', marker='o')
    y_r=fm.eval(nu=x,get_model=True)/86400
    plt.plot( x ,  y_r,'--',label=r'$t^{obs}_{rise}$ best fit')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r' $\beta_{exp}$  (v/c)')
    plt.ylabel(' $t_{rise}^{obs}$  (d)')
    plt.legend()
    R0=fm.components.ExpTrise.parameters.R0.val
    print('R0 fit %e'%R0)
    print('R0 fit *delta %e'%(R0*delta))
    p=eval_p(fm.components.ExpTrise.parameters.phi.val,m_B=1)
    print('p fit  %e'%(p))

    print('------ Decay')
    data=Data(n_rows=beta_exp.size)
    data.set_field('x' ,beta_exp)
    data.set_field('y' ,t_decay_days*86400)
    data.set_field('dy',t_decay_days_err*86400)

    fit_func=ExpTdec_vexp()
        
    fm=FitModel(analytical=fit_func,name='test')
    fm.nu_min_fit=beta_exp.min()
    fm.nu_max_fit=beta_exp.max()
    R_tr_input=(R0_cm)/delta
    fm.components.ExpTdec_vexp.parameters.R0.val=R_tr_input
    fm.components.ExpTdec_vexp.parameters.R0.fit_range=[R_tr_input/10,R_tr_input*10]
    fm.components.ExpTdec_vexp.parameters.nu_1.val=1.5E10
    fm.components.ExpTdec_vexp.parameters.nu_1.frozen=True
   
    fm.components.ExpTdec_vexp.parameters.nu_0.val=nu0
    fm.components.ExpTdec_vexp.parameters.nu_0.fit_range=[1E9,1E12]
    

    bfm,mm=fit_XY(fm,data,x_fit_start=beta_min,x_fit_stop=beta_exp.max(),minimizer='minuit',silent=True)
    mm.show_report()
    
    R0_fit=fm.components.ExpTdec_vexp.parameters.R0.val
    
    print('R_0 %e'%R0_cm)
   
    print('R0_fit %e'%R0_fit)
    print('R0_fit*delta %e'%(R0_fit*delta))
    p=eval_p(fm.components.ExpTdec_vexp.parameters.phi.val,m_B=fm.components.ExpTdec_vexp.parameters.m_B.val)
    print('p fit  %e'%(p))

    f_tdec=plt.figure(dpi=100)
    y_d=fm.eval(nu=x,get_model=True)/86400 
    plt.errorbar( beta_exp , t_decay_days ,yerr=t_decay_days_err, ls='', marker='o')
    plt.plot( x , y_d ,'--',label=r' $t^{obs}_{decay}$ best fit')
    


    print('------ beta cooling -----------')
    fit_func=ExpTdec_vexp_beta_S()
        
    fm=FitModel(analytical=fit_func,name='test')
    fm.nu_min_fit=beta_exp.min()
    fm.nu_max_fit=beta_exp.max()
    R_tr_input=(R0_cm)/delta
    fm.components.ExpTdec_vexp_beta_S.parameters.R0.val=R_tr_input
    fm.components.ExpTdec_vexp_beta_S.parameters.R0.fit_range=[R_tr_input/10,R_tr_input*10]
    fm.components.ExpTdec_vexp_beta_S.parameters.nu_1.val=1.5E10
    fm.components.ExpTdec_vexp_beta_S.parameters.nu_1.frozen=True
   
    fm.components.ExpTdec_vexp_beta_S.parameters.nu_0.val=nu0
    fm.components.ExpTdec_vexp_beta_S.parameters.nu_0.fit_range=[1E9,1E12]
    

    bfm,mm=fit_XY(fm,data,x_fit_start=beta_min,x_fit_stop=beta_exp.max(),minimizer='minuit',silent=True)
    mm.show_report()
    
    R0_fit=fm.components.ExpTdec_vexp_beta_S.parameters.R0.val
   
    print('R_0 %e'%R0_cm)
   
    print('R0_fit %e'%R0_fit)
    print('R0_fit*delta %e'%(R0_fit*delta))
    p=eval_p(fm.components.ExpTdec_vexp_beta_S.parameters.phi.val,m_B=fm.components.ExpTdec_vexp_beta_S.parameters.m_B.val)
    print('p fit  %e'%(p))

    y_d=fm.eval(nu=x,get_model=True)/86400 

    plt.plot( x , y_d ,'--',label=r' $t^{obs}_{decay^{*}}$ (Eq. 23) best fit')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r'$\beta_{exp}$  (v/c)')
    plt.ylabel(r'$t^{obs}_{decay}$  (d)')
    plt.legend()
    
    print('------ ExpDeltaT_vexp')
    data=Data(n_rows=beta_exp.size)
    data.set_field('x' ,beta_exp)
    data.set_field('y' ,delta_T_days*86400)
    data.set_field('dy',delta_T_days_err*86400)
   
    fit_func=ExpDeltaT_vexp()
    
    fm=FitModel(analytical=fit_func,name='test')
    fm.nu_min_fit=beta_exp.min()
    fm.nu_max_fit=beta_exp.max()
    

    R_tr_input=(R0_cm)/delta
    fm.components.DeltaT.parameters.R0.val=R_tr_input
    fm.components.DeltaT.parameters.R0.fit_range=[R_tr_input/10,R_tr_input*10]

    fm.components.DeltaT.parameters.t_exp.val=t_exp_sym/delta
    fm.components.DeltaT.parameters.t_exp.fit_range=[t_exp_sym/delta/10,t_exp_sym/delta*10]

    fm.components.DeltaT.parameters.nu_1.val=1.5E10
    fm.components.DeltaT.parameters.nu_1.frozen=True
   
    fm.components.DeltaT.parameters.nu_0.val=nu0
    fm.components.DeltaT.parameters.nu_0.fit_range=[1E9,1E12]
    
    bfm,mm=fit_XY(fm,data,x_fit_start=beta_min,x_fit_stop=beta_exp.max(),minimizer='minuit',silent=True)
    mm.show_report()
    
    R0=fm.components.DeltaT.parameters.R0.val
    t_exp=fm.components.DeltaT.parameters.t_exp.val
    print('R_0 %e'%R0_cm)
    
    print('R0_fit%e'%R0)
    print('R0_fit*delta %e'%(R0*delta))
    print('t_exp %e'%t_exp)
    print('t_exp_sym %e'%(t_exp_sym))
    print('t_exp_sym/delta %e'%(t_exp_sym/delta))
    p=eval_p(fm.components.DeltaT.parameters.phi.val,m_B=1)
    print('p fit  %e'%(p))


    f_delta_t=plt.figure(dpi=100)
    plt.errorbar(beta_exp,delta_T_days,yerr=delta_T_days_err, ls='', marker='o')
    plt.axhline( t_exp/86400,ls='-',label=r'$ t_{exp}^{obs}$ best fit ',c='g')
    plt.axhline( t_exp_sym/86400/delta,ls='--',label='$ t_{exp}^{obs}$ sim',c='r')
    plt.plot( x,  fm.eval(nu=x,get_model=True)/86400 ,'--',label=r'$\Delta^{obs}_t$ best fit')
    plt.xscale("log")
    plt.yscale("log")
    
    plt.xlabel(r' $\beta_{exp}$ (v/c)')
    plt.ylabel(' $\Delta_t^{obs}$ (d)')
    plt.legend()
    
    
    
    print('------')
    
    
   
    f_sp=plt.figure(dpi=100)
    
    ax1=f_sp.add_subplot()
    
    y=t_rise_days/t_decay_days
    y_err=y*np.sqrt((t_rise_days_err/t_rise_days)**2+(t_decay_days_err/t_rise_days))
    ax1.errorbar(beta_exp,y,yerr=y_err,marker='o',ls='',label=r'$t_{rise}^{obs}/t_{decay}^{obs}$')

    ax1.plot(x,y_r/y_d,'--',label=r'$t_{rise}^{obs}/t_{decay^*}^{obs}$ best fit model')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$t_{rise}^{obs}/t_{decay}^{obs}$')   
    ax1.set_xlabel(r' $\beta_{exp}$ (v/c)')  
    ax1.legend()
    f_sp.tight_layout()



    return f_tdec,f_trise,f_delta_t,f_sp


class ExpDeltaT_freq(Model):
    """
    Class to handle function for growth model
    """
    
    def __init__(self,nu_size=100,**keywords):
        """
        """
     
        super(ExpDeltaT_freq,self).__init__(  **keywords)
        self.name='DeltaT'
        self.nu_min=-100
        self.nu_max=100
        self.parameters = ModelParameterArray()      
        #self.R0=1
        self.parameters.add_par(AnalyticalParameter(self,name='R0',par_type='',val=1,val_min=0,val_max=None,units='cm'))        
        #self.t_exp=1
        self.parameters.add_par(AnalyticalParameter(self,name='t_exp',par_type='',val=1,val_min=0.,val_max=None,units='s'))
        #self.beta_exp=0.5
        self.parameters.add_par(AnalyticalParameter(self,name='beta_exp',par_type='',val=0.5,val_min=0.,val_max=1,units='c'))
        self.parameters.add_par(AnalyticalParameter(self,name='nu_0',par_type='',val=1E10,val_min=1E6,val_max=1E15,units='Hz'))
        self.parameters.add_par(AnalyticalParameter(self,name='phi',par_type='',val=0.5,val_min=-2,val_max=2,units=''))

    def lin_func(self,nu):
        a=(1)*(self.parameters.R0.val/(self.parameters.beta_exp.val*3E10))
        #phi=(self.parameters.p.val+4)/(2*self.parameters.m_B.val*(self.parameters.p.val+2)+4)      
        phi=self.parameters.phi.val 
        a=a*((self.parameters.nu_0.val/nu)**(phi)-1)

        a[a<=0]=0
        return  self.parameters.t_exp.val + a





class ExpTrise_nu(Model):
    """
    Class to handle function for growth model
    """
    
    def __init__(self,nu_size=100,**keywords):
        """
        """
     
        super(ExpTrise_nu,self).__init__(  **keywords)
        self.name='ExpTrise'
        self.nu_min=-100
        self.nu_max=100
        self.parameters = ModelParameterArray()      
        #self.R0=1
        self.parameters.add_par(AnalyticalParameter(self,name='R0',par_type='',val=1,val_min=0,val_max=None,units='cm'))        
        #self.t_exp=1
        #self.beta_exp=0.5
        self.parameters.add_par(AnalyticalParameter(self,name='beta_exp',par_type='',val=0.5,val_min=0.,val_max=1,units='c'))
        self.parameters.add_par(AnalyticalParameter(self,name='nu_0',par_type='',val=1E10,val_min=1E6,val_max=1E15,units='Hz'))
        self.parameters.add_par(AnalyticalParameter(self,name='phi',par_type='',val=0.5,val_min=-2,val_max=2,units=''))
        #self.parameters.add_par(AnalyticalParameter(self,name='m_B',par_type='',val=1.0,val_min=0.5,val_max=2.5,units=''))

    def lin_func(self,nu):
        a=(1/3)*(self.parameters.R0.val/(self.parameters.beta_exp.val*3E10))
        #phi=(self.parameters.p.val+4)/(2*self.parameters.m_B.val*(self.parameters.p.val+2)+4)
        phi=self.parameters.phi.val       
        a=a*((self.parameters.nu_0.val/nu)**(phi)-1)

        a[a<=0]=0
        return a



class ExpDec_nu(Model):
    """
    Class to handle function for growth model
    """
    
    def __init__(self,nu_size=100,**keywords):
        """
        """
     
        super(ExpDec_nu,self).__init__(  **keywords)
        self.name='ExpTdec'
        self.nu_min=-100
        self.nu_max=100
        self.parameters = ModelParameterArray()      
        #self.R0=1
        self.parameters.add_par(AnalyticalParameter(self,name='R0',par_type='',val=1,val_min=0,val_max=None,units='cm'))        
        #self.t_exp=1
        #self.beta_exp=0.5
        self.parameters.add_par(AnalyticalParameter(self,name='beta_exp',par_type='',val=0.5,val_min=0.,val_max=1,units='c'))
        self.parameters.add_par(AnalyticalParameter(self,name='nu_0',par_type='',val=1E10,val_min=1E6,val_max=1E15,units='Hz'))
        self.parameters.add_par(AnalyticalParameter(self,name='phi',par_type='',val=0.5,val_min=-2,val_max=2,units=''))
        self.parameters.add_par(AnalyticalParameter(self,name='m_B',par_type='',val=1.0,val_min=0.5,val_max=2.5,units=''))

    def lin_func(self,nu):
        a=(self.parameters.R0.val/(self.parameters.beta_exp.val*3E10))
        a=a/(2*self.parameters.m_B.val)
        #phi=(self.parameters.p.val+4)/(2*self.parameters.m_B.val*(self.parameters.p.val+2)+4)
        phi=self.parameters.phi.val       
        a=a*(self.parameters.nu_0.val/nu)**(phi)
        return a


def gamma_radio_delay_analysis_vs_freq(freq_radio,
                                       t_decay_days,
                                        t_decay_days_err,
                                        t_rise_days,
                                        t_rise_days_err,
                                        delta_T_days,
                                        delta_T_days_err,
                                        t_decay_gamma_days,
                                        R0_cm,
                                        t_exp_sym,
                                        delta,
                                        beta_exp):
    print('------ ExpTrise')

    data=Data(n_rows=freq_radio.size)
    data.set_field('x' ,freq_radio)
    data.set_field('y' ,t_rise_days*86400)
    data.set_field('dy',t_rise_days_err*86400)
   
    x=np.linspace(freq_radio.min()*0.9,freq_radio.max()*5,100)
    nu0=1E11

    beta_range=[0,0.3]
    fit_range=[1E10,1E11]
    fit_func=ExpTrise_nu()
        
    fm=FitModel(analytical=fit_func,name='test')
    fm.nu_min_fit=freq_radio.min()
    fm.nu_max_fit=freq_radio.max()
    R_tr_input=(R0_cm)/delta
    fm.components.ExpTrise.parameters.R0.val=R_tr_input
    fm.components.ExpTrise.parameters.R0.fit_range=[R_tr_input/10,R_tr_input*10]
    fm.components.ExpTrise.parameters.beta_exp.val=beta_exp
    fm.components.ExpTrise.parameters.beta_exp.fit_range=beta_range
    fm.components.ExpTrise.parameters.beta_exp.frozen=False
    fm.components.ExpTrise.parameters.nu_0.val=nu0
    fm.components.ExpTrise.parameters.nu_0.fit_range=fit_range
    
    bfm,mm=fit_XY(fm,data,x_fit_start=freq_radio.min(),x_fit_stop=freq_radio.max(),minimizer='minuit',silent=True)
    mm.show_report()
    
    f_trise=plt.figure(dpi=100)
    plt.errorbar( freq_radio , t_rise_days,yerr=t_rise_days_err, ls='', marker='o')
    y_r=fm.eval(nu=x,get_model=True)/86400
    nu_0_bf= fm.components.ExpTrise.parameters.nu_0.val
    plt.plot( x ,  y_r,'--',label=r'$t^{obs}_{rise}$ best  fit')
    plt.axvline( 180*1E9,ls='--',label=r'$ \nu^{0, \rm obs}_{SSA}$ sym.',c='red')
    plt.axvline( nu_0_bf,ls='--',label=r'$\nu^{0, \rm obs}_{SSA}$ best fit ',c='g')
    plt.axvspan(nu_0_bf-fm.components.ExpTrise.parameters.nu_0.best_fit_err, 
                nu_0_bf+fm.components.ExpTrise.parameters.nu_0.best_fit_err, alpha=0.1, color='g')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r' $\nu_{obs}$ (Hz)')
    plt.ylabel(' $t_{rise}^{obs}$  (d)')
    plt.legend()
    R0_t_rise=fm.components.ExpTrise.parameters.R0 .val
  
    print('R_0  %e'%R0_cm)
    print('R0 fit %e'%(R0_t_rise))
    print('R0 fit *delta %e'%(R0_t_rise*delta))
    
    print('------ Decay')
    data=Data(n_rows=freq_radio.size)
    data.set_field('x' ,freq_radio)
    data.set_field('y' ,t_decay_days*86400)
    data.set_field('dy',t_decay_days_err*86400)


    fit_func=ExpDec_nu()
        
    fm=FitModel(analytical=fit_func,name='test')
    fm.nu_min_fit=freq_radio.min()
    fm.nu_max_fit=freq_radio.max()
    R_tr_input=(R0_cm)/delta
    fm.components.ExpTdec.parameters.R0.val=R_tr_input
    fm.components.ExpTdec.parameters.R0.fit_range=[R_tr_input/10,R_tr_input*10]
    fm.components.ExpTdec.parameters.beta_exp.val=beta_exp
    fm.components.ExpTdec.parameters.beta_exp.fit_range=beta_range
    fm.components.ExpTdec.parameters.beta_exp.frozen=False
    fm.components.ExpTdec.parameters.nu_0.val=nu0
    fm.components.ExpTdec.parameters.nu_0.fit_range=fit_range

    bfm,mm=fit_XY(fm,data,x_fit_start=freq_radio.min(),x_fit_stop=freq_radio.max(),minimizer='minuit',silent=True)
    mm.show_report()
    R_0_fit=fm.components.ExpTdec.parameters.R0.val
    
    print('R_0  %e'%R0_cm)
    print('R_0 fit %e'%R_0_fit)
    print('R_0_fit*delta %e'%(R_0_fit*delta))
   

    f_tdec=plt.figure(dpi=100)
    plt.errorbar( freq_radio , t_decay_days ,yerr=t_decay_days_err, ls='', marker='o')
    y_d=fm.eval(nu=x,get_model=True)/86400 
    plt.plot( x ,y_d  ,'--',label=r' $t^{obs}_{decay}$ best fit')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r' $\nu_{obs}$ (Hz)')
    plt.ylabel(r'$t^{obs}_{decay}$  (d)')
    plt.legend()
    
    print('------ ExpDeltaT_vexp')
    data=Data(n_rows=freq_radio.size)
    data.set_field('x' ,freq_radio)
    data.set_field('y' ,delta_T_days*86400)
    data.set_field('dy',delta_T_days_err*86400)
   
    fit_func=ExpDeltaT_freq()
    
    fm=FitModel(analytical=fit_func,name='test')
    fm.nu_min_fit=freq_radio.min()
    fm.nu_max_fit=freq_radio.max()
    
    R_tr_input=(R0_cm)/delta
    fm.components.DeltaT.parameters.R0.val=R0_t_rise
    fm.components.DeltaT.parameters.R0.fit_range=[R_tr_input/10,R_tr_input*10]
    
    fm.components.DeltaT.parameters.t_exp.val=t_exp_sym/delta
    fm.components.DeltaT.parameters.t_exp.fit_range=[t_exp_sym/delta/2,t_exp_sym/delta*2]
    fm.components.DeltaT.parameters.nu_0.val=nu0
    
    fm.components.DeltaT.parameters.beta_exp.val=beta_exp
    fm.components.DeltaT.parameters.beta_exp.fit_range=beta_range
    fm.components.DeltaT.parameters.beta_exp.frozen=False
   
    fm.components.DeltaT.parameters.nu_0.fit_range=fit_range

    bfm,mm=fit_XY(fm,data,x_fit_start=freq_radio.min(),x_fit_stop=freq_radio.max(),minimizer='minuit',silent=True)
    mm.show_report()
    R_0_fit=fm.components.DeltaT.parameters.R0.val
    t_exp=fm.components.DeltaT.parameters.t_exp.val
    print('R_0  %e'%R0_cm)
    
    print('R_0_fit %e'%R_0_fit)
    print('R_0_fit*delta %e'%(R_0_fit*delta))
    print('t_exp %e'%t_exp)
    print('t_exp d %e'%(t_exp/86400))
    print('t_exp_sym %e'%(t_exp_sym/delta))
    
    f_delta_t=plt.figure(dpi=100)
    plt.axhline( t_exp/86400,ls='-',label='$ t_{exp}^{obs}~ best fit $',c='g')
    plt.axhline( t_exp_sym/86400/delta,ls='--',label='$ t_{exp}^{obs} ~sim. $',c='r')
    plt.errorbar(freq_radio,delta_T_days,yerr=delta_T_days_err, ls='', marker='o')
    plt.plot( x,  fm.eval(nu=x,get_model=True)/86400 ,'--',label=r'$\Delta_t = t^{obs}_{exp}$ best fit')
    
   
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r' $\nu_{obs}$ (Hz)')
    plt.ylabel(' $\Delta_t^{obs}$ (d)')
    plt.legend()
    
    f_delta_t_1=plt.figure(dpi=100)
    plt.plot( x,  fm.eval(nu=x,get_model=True)/86400 ,'--',)
    plt.plot( x,  fm.eval(nu=x,get_model=True)/86400-t_exp/86400 ,'--',)
    plt.axhline( t_exp/86400,ls='-',label='$ t_{exp}~ best fit. $',c='g')
    plt.xscale("log")
    plt.yscale("log")
    print('------')
   
    
    f_sp=plt.figure(dpi=100)
   
    ax1=f_sp.add_subplot()
   
    y=t_rise_days/t_decay_days
    y_err=y*np.sqrt((t_rise_days_err/t_rise_days)**2+(t_decay_days_err/t_rise_days))
    ax1.errorbar(freq_radio,y,yerr=y_err,marker='o',ls='',label=r'$t_{rise}^{obs}/t_{decay}^{obs}$')

    ax1.plot(x,y_r/y_d,'--',label=r'$t_{rise}^{obs}/t_{decay}^{obs}$ model')
    ax1.axvline( nu_0_bf,ls='--',label=r'$\nu^{0, \rm obs}_{SSA}$  best fit  model',c='g')
    ax1.set_xscale('log')

    ax1.set_ylabel(r'$t_{rise}^{obs}/t_{decay}^{obs}$')       
    ax1.set_xlabel(r' $\nu_{obs}$ (Hz)')  
    ax1.legend()
    f_sp.tight_layout()
    return f_tdec,f_trise,f_delta_t,f_sp

def plot_lcs(exp_lcs,lcs_names,flare_lcs=None):
    lc_plot_list=[None]*len(lcs_names)
       
    with open(exp_lcs, 'rb') as f:
            exp_lcs=pickle.load(f)  
    
    if flare_lcs is not None:
        with open(flare_lcs, 'rb') as f:
            lcs_flare=pickle.load(f)
        for ID,f in enumerate(lcs_names):
            lc_plot_list[ID]=merge_lc(lcs_flare[lcs_names[ID]],exp_lcs[lcs_names[ID]])
    else:
        for ID,f in enumerate(lcs_names):
            lc_plot_list[ID]=exp_lcs[lcs_names[ID]]
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])


    
    y_l=[None]*len(lcs_names)
    y_l_err=[None]*len(lcs_names)
    t_l=[None]*len(lcs_names)
    y_label=[None]*len(lcs_names)
    labels=[None]*len(lcs_names)
    
    fig = plt.figure(constrained_layout=True)
    fig_h=3*len(lc_plot_list)
    fig_w=12
    gs_w=7
    d_w=0

    fig.set_size_inches(fig_w,fig_h)


    gs = fig.add_gridspec(ncols=1, nrows=len(lcs_names))
    ax=[None]*len(lc_plot_list)
    for ID,f in enumerate(lc_plot_list):
        ax[ID]= fig.add_subplot(gs[ID,0])
    
    plt.tight_layout()

    for ID,f in enumerate(lc_plot_list):
       
        labels[ID]=lc_plot_list[ID].meta['name']

        
        t_l[ID]=lc_plot_list[ID]['time'].to('d').value
        y_l[ID]=lc_plot_list[ID]['flux']
        if 'e_flux' in lc_plot_list[ID].colnames:
            y_l_err[ID]=lc_plot_list[ID]['e_flux']
        else:
            y_l_err[ID]=np.zeros(len(lc_plot_list[ID]))
        
        y_label[ID]=lc_plot_list[ID]['flux'].unit
        
        ax[ID].errorbar(x=t_l[ID],y=y_l[ID],yerr=0,label=labels[ID].replace('_',' '),marker=None,ms=1,lw=1.,linestyle='-')
        ax[ID].legend() 
        ax[ID].set_ylabel(y_label[0])
        if flare_lcs is not None:
            ax[ID].plot(lcs_flare[lcs_names[ID]]['time'].to('d').value,lcs_flare[lcs_names[ID]]['flux'],label='flare',ls='--',c='r',lw=2)
    

        ax[ID].legend()
  
    
    ax[-1].set_ylabel(y_label[1])
    ax[-1].set_xlabel(r'$t^{\rm obs}$ (d)')
    

    return fig


def plot_lcs_single_panel(exp_lcs,lcs_names,flare_lcs=None):
    lc_plot_list=[None]*len(lcs_names)
       
    with open(exp_lcs, 'rb') as f:
            exp_lcs=pickle.load(f)  
    
    if flare_lcs is not None:
        with open(flare_lcs, 'rb') as f:
            lcs_flare=pickle.load(f)
        for ID,f in enumerate(lcs_names):
            lc_plot_list[ID]=merge_lc(lcs_flare[lcs_names[ID]],exp_lcs[lcs_names[ID]])
    else:
        for ID,f in enumerate(lcs_names):
            lc_plot_list[ID]=exp_lcs[lcs_names[ID]]
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])


    
    y_l=[None]*len(lcs_names)
    y_l_err=[None]*len(lcs_names)
    t_l=[None]*len(lcs_names)
    y_label=[None]*len(lcs_names)
    labels=[None]*len(lcs_names)
    
    fig = plt.figure(constrained_layout=True)
    fig_h=6
    fig_w=12
    gs_w=7
    d_w=0

    fig.set_size_inches(fig_w,fig_h)


    gs = fig.add_gridspec(ncols=1, nrows=1)
    #ax=[None]*len(lc_plot_list)
    #for ID,f in enumerate(lc_plot_list):
    ax= fig.add_subplot(gs[0,0])
    
    plt.tight_layout()

    for ID,f in enumerate(lc_plot_list):
       
        labels[ID]=lc_plot_list[ID].meta['name']

        
        t_l[ID]=lc_plot_list[ID]['time'].to('d').value
        y_l[ID]=lc_plot_list[ID]['flux']
        if 'e_flux' in lc_plot_list[ID].colnames:
            y_l_err[ID]=lc_plot_list[ID]['e_flux']
        else:
            y_l_err[ID]=np.zeros(len(lc_plot_list[ID]))
        
        y_label[ID]=lc_plot_list[ID]['flux'].unit
        
        ax.errorbar(x=t_l[ID],y=y_l[ID],yerr=0,label=labels[ID].replace('_',' '),marker=None,ms=1,lw=1.,linestyle='-')
        ax.legend() 
        ax.set_ylabel(y_label[0])
        if flare_lcs is not None:
            ax.plot(lcs_flare[lcs_names[ID]]['time'].to('d').value,lcs_flare[lcs_names[ID]]['flux'],label=None,ls='--',c='r',lw=2)
    

        ax.legend()
  
    
    ax.set_ylabel(y_label[1])
    ax.set_xlabel(r'$t^{\rm obs}$ (d)')
    

    return fig