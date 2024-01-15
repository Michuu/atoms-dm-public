##
using PyPlot
using QuantumOptics
using LinearAlgebra
using Arpack
using SpecialFunctions
using SharedArrays
##
ion()
pygui(true)
##
function χ(Delta0_1, Delta0_2, Delta0_3, Delta0_4, power_factor, Omega_mw)

    b = NLevelBasis(5)
    s_gp = transition(b, 1, 2)
    s_pr1 = transition(b, 2, 3)
    s_r1r2 = transition(b, 3, 4)
    s_r2d = transition(b, 4, 5)
    s_dp = transition(b, 5, 2)

    s_r1g = transition(b,3,1)
    s_r2g = transition(b,4,1)
    s_dg = transition(b,5,1)

    proj_g= transition(b,1,1)
    proj_p = transition(b,2,2)
    proj_r1  = transition(b,3,3)
    proj_r2  = transition(b,4,4)
    proj_d = transition(b,5,5)

    Gamma_tt=2*π*5
    Gamma_r1g=Gamma_tt
    Gamma_r2g=Gamma_tt
    Gamma_dg=Gamma_tt

    Gamma_p = 2*π*6 + Gamma_tt
    Gamma_r1 = 2*π*.01
    Gamma_r2 = 2*π*.01
    Gamma_d = 2*π*.9
    Gamma_deph=2π*4

    #power_factor=0.6
    Omega_p=2*π*4 * power_factor
    Omega_c=2*π*20.8* power_factor
    #Omega_mw=2*π*0.1
    Omega_d =2*π*18* power_factor
    Omega_s = 2π*0

    k1=2*π/780e-3
    k2=2*π/480e-3
    k3=2*π/2100
    k4=2*π/1260e-3

    m=1.41e-25
    kB=1.38e-23
    T=273 + 42
    J = [sqrt(Gamma_p) * (s_gp), sqrt(Gamma_r1) * (s_pr1), 
    sqrt(Gamma_r2) * dagger(s_r2d), sqrt(Gamma_d) * dagger(s_dp), 
    sqrt(Gamma_r1g) * dagger(s_r1g), sqrt(Gamma_r2g)* dagger(s_r2g), 
    sqrt(Gamma_dg)* dagger(s_dg), sqrt(1e-14) * transition(b,2,1), sqrt(Gamma_deph)*proj_r1, sqrt(Gamma_deph)*proj_r2]
    Hi =  Omega_p/2 * (s_gp + dagger(s_gp)) + Omega_c/2 * (s_pr1 + dagger(s_pr1)) + Omega_mw/2 * (s_r1r2 + dagger(s_r1r2)) + Omega_d/2 * (s_r2d + dagger(s_r2d)) + Omega_s/2 * (s_dp+dagger(s_dp))

    ndgauss=500

    chi=zeros(ComplexF64, (ndgauss))
    #chi_eit=zeros(ComplexF64, (ndgauss))
    #Delta_1s = LinRange(-2*pi*45,2*pi*45)

    v=1
    Delta_1v =  k1*v
    Delta_2v = k1*v - k2*v
    Delta_3v = k1*v - k2*v - k3*v
    Delta_4v = k1*v - k2*v - k3*v + k4*v
    HK =  Delta_1v * proj_p + Delta_2v * proj_r1 + Delta_3v * proj_r2 + Delta_4v * proj_d

    LK=QuantumOpticsBase.liouvillian(HK,[])
    LK=Matrix(LK.data)
    L0=QuantumOpticsBase.liouvillian(Hi,J,)
    L0=Matrix(L0.data)

    reshape(diag(L0),(5,5))
    vlim=255
    grange=LinRange(-vlim,vlim,ndgauss)

    dv=2*vlim/ndgauss
    norm=1/(sqrt(2π*kB*T/m)) * dv #erf(vlim*sqrt(m/(2*kB*T)))) 

    Delta_1 =  Delta0_1
    Delta_2 = Delta0_2 + Delta_1 
    Delta_3 = Delta0_3 + Delta_2 
    Delta_4 = Delta0_4 + Delta_3
    Hb=Delta_1 * proj_p + Delta_2 * proj_r1 + Delta_3 * proj_r2 + Delta_4 * proj_d
    L1=QuantumOpticsBase.liouvillian(Hb,[],)

    L1=Matrix(L1.data) + L0

    F=eigen(LK,L1)
    P=F.vectors

    D=Diagonal(F.values)
    rho0=eigs(L1,nev=1,which=:SM)[2]

    for (i, v) in enumerate(grange)
        rho=P*inv(-v*D+I)*(P\rho0)
        rho=reshape(rho,(5,5))
        rho=rho/tr(rho)
        chi[i] = rho[5,2]* exp(-m*v^2/(2*kB*T)) * norm
        #chi_eit[i,j] = rho[1,2]* exp(-m*v^2/(2*kB*T)) * norm
    end

    ch=transpose(sum(chi,dims=1))
    return ch[1]
end
##
Delta0_1 = 2*π*0
Delta0_2 = -2*π*30
Delta0_3 = 2*π*30
Delta0_4 = 2*π*0
##
len=150
res=zeros(ComplexF64, (len))
Ommws= exp10.(range(-3, stop=3, length=len))
for (i, v) in enumerate(Ommws)
    try
        res[i]=χ(1,2π*30,-2π*30,0,0.5, v)
    catch
        res[i] = 0
    end
end
##
χ(5,0,0,0, 0.6, 1)
##
zeros(10,Com)
##
plot(broadcast(real,res))
plot(broadcast(imag,res))
##
plot(Ommws, broadcast(abs2,res))

##
for dl in (0,20)
    len=150
    res=zeros(ComplexF64, (len))
    delta= 2π* range(start=-50,stop=50,length=len)
    for (i, v) in enumerate(delta)
        try
            res[i]=χ(v,-2π*dl,2π*(dl),0,0.6, 2π*0)
        catch
            res[i] = 0
        end
    end
    plot(delta/(2π), broadcast(real,res),label=dl)
end
legend()
##
for dl in (0,20)
    lend=150
    len=15
    res=zeros(ComplexF64, (lend,len))
    delta= 2π* range(start=-40,stop=40,length=lend)
    for (j, w) in enumerate(delta)
        pf = range(start=0.1,stop=1,length=len)
        #pf= range(start=0.0,stop=1,length=len)
        for (i, v) in enumerate(pf)
            try
                res[j,i]=χ(w,2π*dl,-2π*(dl),0,v, 2π*10) * sqrt(2*sqrt(-log(v)/π))
            catch
                res[j,i] = 0
            end
        end
        #plot(pf, broadcast(imag,res),label=dl)
    end
    plot(delta/(2π), broadcast(abs2,sum(res,dims=2)),label=dl)
    #plot(delta/(2π), broadcast(abs2,sum(res,dims=2)) .* sinc.(40*broadcast(real,sum(phmf,dims=2))).^2,label=dl)
    
end
legend()
##
phmf=res
##
plot(delta/(2π), broadcast(abs2,sum(res,dims=2)) .* sinc.(90*broadcast(real,sum(phmf,dims=2))).^2)
##
lend=25
lenf=25
delta2=2π* range(start=-60,stop=60,length=lenf)
ress=zeros(ComplexF64, (lenf,lend))
for (m, dl) in enumerate(delta2)
    len=15
    res=zeros(ComplexF64, (lend,len))
    delta= 2π* range(start=-40,stop=40,length=lend)
    for (j, w) in enumerate(delta)
        pf = range(start=0.1,stop=1,length=len)
        #pf= range(start=0.0,stop=1,length=len)
        for (i, v) in enumerate(pf)
            try
                res[j,i]=χ(-0,(w+0.0*dl),(dl)-(w+0.0*dl),-(dl),v, 2π*0.1) * sqrt(2*sqrt(-log(v)/π))
            catch
                res[j,i] = 0
            end
        end
        #plot(pf, broadcast(imag,res),label=dl)
    end
    #plot(delta/(2π), broadcast(abs2,sum(res,dims=2)),label=dl)
    ress[m,:] = broadcast(abs2,sum(res,dims=2))
    #plot(delta/(2π), broadcast(abs2,sum(res,dims=2)) .* sinc.(40*broadcast(real,sum(phmf,dims=2))).^2,label=dl)
    
end
#legend()

pcolormesh(broadcast(abs,ress))

##
lend=50
lenf=2
delta2=2π* range(start=-0,stop=0,length=lenf)
ress=zeros(ComplexF64, (lenf,lend))
for (m, dl) in enumerate(delta2)
    len=10
    res=zeros(ComplexF64, (lend,len))
    delta= 2π* range(start=-40,stop=40,length=lend)
    for (j, w) in enumerate(delta)
        pf = range(start=0.1,stop=1,length=len)
        #pf= range(start=0.0,stop=1,length=len)
        for (i, v) in enumerate(pf)
            try
                res[j,i]=χ(-0,16*2π,-16*2π+w,-w,v, 2π*0.1) * sqrt(2*sqrt(-log(v)/π))
            catch
                res[j,i] = 0
            end
        end
        #plot(pf, broadcast(imag,res),label=dl)
    end
    plot(delta/(2π), broadcast(abs2,sum(res,dims=2)),label=dl)
    ress[m,:] = broadcast(abs2,sum(res,dims=2))
    #plot(delta/(2π), broadcast(abs2,sum(res,dims=2)) .* sinc.(40*broadcast(real,sum(phmf,dims=2))).^2,label=dl)
    
end
legend()
##