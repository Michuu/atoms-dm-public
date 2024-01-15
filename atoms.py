# %%
from julia import Main
import numpy as np
import matplotlib.pyplot as plt
#%%
Main.eval('include("./Atom5lvl.jl")')
# %%
rho = Main.Atom5lvl.ρ
# %%
π = 3.14
Omega_p=2*π*4
Omega_c=2*π*20.8
Omega_mw=2*π*0.1
Omega_d =2*π*18
Omega_s = 2*π*0

# deph is for the Rydberg-state dephasing, tt is for the transit time broadening (atom losses)
Gamma_tt=2*π*5
Gamma_deph=2*π*4
#%%
r0=rho(0, 0, 0, 0, Omega_mw, Omega_p, Omega_c, Omega_d, Omega_s, Gamma_tt, Gamma_deph)
#%%
%timeit rho(0, 0, 0, 0, Omega_mw, Omega_p, Omega_c, Omega_d, Omega_s, Gamma_tt, Gamma_deph)

# %% eit + conv vs probe detuning
fig, ax = plt.subplots(2,1,sharex=True)
χ = lambda *args: rho(*args)[4,1]

π = np.pi
Omega_p=2*π*8
Omega_c=2*π*22.0
Omega_mw=2*π*10
Omega_d =2*π*17
Omega_s = 2*π*0

Delta1 =2*π*100
Gamma_tt=2*π*3
Gamma_deph=2*π*0
data = []
for m, dl in enumerate([0,20]):
    lend=50 
    len=10
    res_conv=np.zeros((lend,len),dtype=np.complex64)
    res_eit=np.zeros((lend,len),dtype=np.complex64)
    delta= 2*π* np.linspace(-40,40,lend)
    # the external loop creates the vector over detunings
    for (j, w) in enumerate(delta):
        pf = np.linspace(0.1,1,len) # pf casually means "power factor" here and is a factor to the Rabi frequencies to account for the Gaussian beam profile
        #pf= range(start=0.0,stop=1,length=len)
        fac=0
        # the internal loop averages over intensities
        for (i, v) in enumerate(pf):
            try:
                ρ = rho(w+Delta1,2*π*dl-Delta1*(780/480),-2*π*dl,Delta1*(780/1250), Omega_mw, Omega_p*v, Omega_c*v, Omega_d*v, Omega_s, Gamma_tt, Gamma_deph) * np.sqrt(2*np.sqrt(-np.log(v)/π))
                #if np.trace(ρ) != 0:
                #    ρ /= np.trace(ρ)
                fac+= np.sqrt(2*np.sqrt(-np.log(v)/π))
                res_conv[j,i]=ρ[4,4]
                res_eit[j,i]=ρ[0,1]
            except:
                res_conv[j,i]=0
                res_eit[j,i]=0
        #plot(pf, broadcast(imag,res),label=dl)
    ax[0].plot(delta/(2*π), np.abs(np.sum(res_conv,axis=1)/fac)**2,label=dl)
    ax[1].plot(delta/(2*π), -np.imag(np.sum(res_eit,axis=1)/fac),label=dl)
    data.append(np.abs(np.sum(res_conv,axis=1))**2)
    data.append(-np.imag(np.sum(res_eit,axis=1)))
    #plot(delta/(2π), broadcast(abs2,sum(res,dims=2)) .* sinc.(40*broadcast(real,sum(phmf,dims=2))).^2,label=dl)
data.append(delta)
plt.legend()
#%%
np.save(r"eit+conv",data)
# %% saturation calculation, i.e. conversion power vs microwave power 
π = np.pi
Omega_p=2*π*8
Omega_c=2*π*22.0
Omega_mw=2*π*0.1
Omega_d =2*π*17
Omega_s = 2*π*0

Delta1 =2*π*0
Gamma_tt=2*π*3
Gamma_deph=2*π*0
# only one detuning here
for dl in (16,):
    lend=50
    len=10
    res=np.zeros((lend,len),dtype=np.complex64)
    omegamw= 2*π* np.logspace(-2,2.5,lend)
    # this loop calculates the result for a range of microwave Rabi frequencies
    for (j, w) in enumerate(omegamw):
        pf = np.linspace(0.1,1,len)
        #pf= range(start=0.0,stop=1,length=len)
        # the internal loop averages over intensities
        for (i, v) in enumerate(pf):
            try:
                ρ=rho(0,2*π*dl,-2*π*dl,0, w, Omega_p*v, Omega_c*v, Omega_d*v, Omega_s, Gamma_tt, Gamma_deph) * np.sqrt(2*np.sqrt(-np.log(v)/π))
                res[j,i]=ρ[4,1]
            except:
                res[j,i] = 0
        #plot(pf, broadcast(imag,res),label=dl)
    plt.loglog(10**6 * omegamw/(2*π), np.abs(np.sum(res,axis=1))**2,label=dl)
    #plot(delta/(2π), broadcast(abs2,sum(res,dims=2)) .* sinc.(40*broadcast(real,sum(phmf,dims=2))).^2,label=dl)
plt.legend()
#%%
np.save(r"nasycanie",[np.abs(np.sum(res,axis=1))**2,10**6 * omegamw/(2*π)])
# %% 55d/54f map, i.e. conversion power vs various detunings (2D map)
π = np.pi
Omega_p=2*π*8
Omega_c=2*π*22.0
Omega_mw=2*π*0.1
Omega_d =2*π*17
Omega_s = 2*π*0

Delta1 =2*π*0
Gamma_tt=2*π*3
Gamma_deph=2*π*0
lend=100
lenf=100
delta2=2*π* np.linspace(-65,65,lenf)
ress=np.zeros((lenf,lend),dtype=np.complex64)
# two loop here for two detunings
for (m, dl) in enumerate(delta2):
    len=10
    res=np.zeros((lend,len),dtype=np.complex64)
    delta= 2*π* np.linspace(-40,40,lend)
    for (j, w) in enumerate(delta):
        pf = np.linspace(0.1,1,len)
        #pf= range(start=0.0,stop=1,length=len)
        # the internal loop averages over intensities
        for (i, v) in enumerate(pf):
            try:
                ρ=rho(0,w,dl-w,-dl,Omega_mw,Omega_p*v, Omega_c*v, Omega_d*v, Omega_s, Gamma_tt, Gamma_deph) * np.sqrt(2*np.sqrt(-np.log(v)/π))
                res[j,i]=ρ[4,1]
            except:
                res[j,i] = 0
        #plot(pf, broadcast(imag,res),label=dl)
    #plot(delta/(2π), broadcast(abs2,sum(res,dims=2)),label=dl)
    ress[m,:] = np.abs(np.sum(res,axis=1))**2
    #plot(delta/(2π), broadcast(abs2,sum(res,dims=2)) .* sinc.(40*broadcast(real,sum(phmf,dims=2))).^2,label=dl)

#%%
X,Y=np.meshgrid(delta/(2*π),delta2/(2*π))
plt.pcolormesh(X,Y,np.abs(ress))
#%%
np.save(r"mapka",[ress,X,Y])

#%% conversion power vs microwave detuning
π = np.pi
Omega_p=2*π*8
Omega_c=2*π*22.0
Omega_mw=2*π*0.01
Omega_d =2*π*17
Omega_s = 2*π*0

Delta1 =2*π*0
Gamma_tt=2*π*3
Gamma_deph=2*π*0

lend=25
#lenf=2
delta2=[0,1]
ress=np.zeros((lenf,lend),dtype=np.complex64)
for (m, dl) in enumerate(delta2):
    len=10
    res=np.zeros((lend,len),dtype=np.complex64)
    delta= 2*π* np.linspace(-140,140,lend)
    for (j, w) in enumerate(delta):
        pf = np.linspace(0.1,1,len)
        #pf= range(start=0.0,stop=1,length=len)
        for (i, v) in enumerate(pf):
            try:
                ρ=rho(-0+Delta1,16*2*π-Delta1*(780/480),-16*2*π+w,-w*dl-8+Delta1*(780/1250),Omega_mw,Omega_p*v, Omega_c*v, Omega_d*v, Omega_s, Gamma_tt, Gamma_deph) * np.sqrt(2*np.sqrt(-np.log(v)/π))
                res[j,i]=ρ[4,1]
            except:
                res[j,i] = 0
        #plot(pf, broadcast(imag,res),label=dl)
    plt.plot(delta/(2*π), np.abs(np.sum(res,axis=1))**2,label=dl)
    ress[m,:] = np.abs(np.sum(res,axis=1))**2
    #plot(delta/(2π), broadcast(abs2,sum(res,dims=2)) .* sinc.(40*broadcast(real,sum(phmf,dims=2))).^2,label=dl)

plt.legend()
# %%
np.save(r"bandwidth",[ress[0],ress[1],delta])