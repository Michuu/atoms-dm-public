module Atom5lvl

using QuantumOptics
using LinearAlgebra
using Arpack
using SharedArrays

export ρ

b = NLevelBasis(5)
s_gp = transition(b, 1, 2)
s_pr1 = transition(b, 2, 3)
s_r1r2 = transition(b, 3, 4)
s_r2d = transition(b, 4, 5)
s_dp = transition(b, 5, 2)

s_r1g = transition(b,3,1)
s_r2g = transition(b,4,1)
s_dg = transition(b,5,1)

proj_p = transition(b,2,2)
proj_r1  = transition(b,3,3)
proj_r2  = transition(b,4,4)
proj_d = transition(b,5,5)

k1=2*π/780e-3
k2=2*π/480e-3
k3=2*π/2100
k4=2*π/1260e-3

m=1.41e-25
kB=1.38e-23
T=273 + 42

v=1
Delta_1v =  k1*v
Delta_2v = k1*v - k2*v
Delta_3v = k1*v - k2*v - k3*v
Delta_4v = k1*v - k2*v - k3*v + k4*v
HK =  Delta_1v * proj_p + Delta_2v * proj_r1 + Delta_3v * proj_r2 + Delta_4v * proj_d

LK=QuantumOpticsBase.liouvillian(HK,[])
LK=Matrix(LK.data)


function ρ(Delta0_1, Delta0_2, Delta0_3, Delta0_4, Omega_mw, Omega_p, Omega_c, Omega_d, Omega_s, Gamma_tt, Gamma_deph, ndgauss=500, vlim=255)
    

    Gamma_r1g=Gamma_tt
    Gamma_r2g=Gamma_tt
    Gamma_dg=Gamma_tt

    Gamma_p = 2*π*6 + Gamma_tt
    Gamma_r1 = 2*π*.01
    Gamma_r2 = 2*π*.01
    Gamma_d = 2*π*.9

    J = [sqrt(Gamma_p) * (s_gp), sqrt(Gamma_r1) * (s_pr1), 
    sqrt(Gamma_r2) * dagger(s_r2d), sqrt(Gamma_d) * dagger(s_dp), 
    sqrt(Gamma_r1g) * dagger(s_r1g), sqrt(Gamma_r2g)* dagger(s_r2g), 
    sqrt(Gamma_dg)* dagger(s_dg), sqrt(1e-14) * transition(b,2,1), sqrt(Gamma_deph)*proj_r1, sqrt(Gamma_deph)*proj_r2]
    Hi =  Omega_p/2 * (s_gp + dagger(s_gp)) + Omega_c/2 * (s_pr1 + dagger(s_pr1)) + Omega_mw/2 * (s_r1r2 + dagger(s_r1r2)) + Omega_d/2 * (s_r2d + dagger(s_r2d)) + Omega_s/2 * (s_dp+dagger(s_dp))


    L0=QuantumOpticsBase.liouvillian(Hi,J,)
    L0=Matrix(L0.data)
    
    grange=LinRange(-vlim,vlim,ndgauss)

    dv=2*vlim/ndgauss
    norm=1/(sqrt(2π*kB*T/m)) * dv

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

    rho_avg = zeros(ComplexF64, (5,5))

    for v in grange
        rho=P*inv(-v*D+I)*(P\rho0)
        rho=reshape(rho,(5,5))
        rho=rho/tr(rho)
        rho_avg += rho * exp(-m*v^2/(2*kB*T)) * norm
    end

    return rho_avg
end
end #module