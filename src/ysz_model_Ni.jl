module ysz_model_Ni

using Printf
using VoronoiFVM
using PyPlot
using DataFrames
using CSV
using LeastSquaresOptim

cur_dir=pwd()
cd("../src")
src_dir=pwd()
cd(cur_dir)

push!(LOAD_PATH,src_dir)
using ysz_model_fitted_parms
const label_ysz_model = ysz_model_fitted_parms


mutable struct YSZParameters <: VoronoiFVM.AbstractData

    # to fit
    A0::Float64   # surface adsorption coefficient [ m^-2 s^-1 ] 
    R0::Float64 # exhange current density [m^-2 s^-1]
    DGA::Float64 # difference of gibbs free energy of adsorbtion  [ J ]
    DGR::Float64 # difference of gibbs free energy of electrochemical reaction [ J ]
    beta::Float64 # symmetry of the reaction
    A::Float64 # activation energy of the reaction
    
    # fixed
    DD::Float64   # diffusion coefficient [m^2/s]
    pO::Float64 # O2 partial pressure [bar]
    T::Float64      # Temperature [K]
    nu::Float64    # ratio of immobile ions, \nu [1]
    nus::Float64    # ratio of immobile ions on surface, \nu_s [1]
    numax::Float64  # max value of \nu 
    nusmax::Float64  # max value of  \nu_s
    x_frac::Float64 # Y2O3 mol mixing, x [%] 
    chi::Float64    # dielectric parameter [1]
    m_par::Float64
    ms_par::Float64

    # known
    vL::Float64     # volume of one FCC cell, v_L [m^3]
    areaL::Float64 # area of one FCC cell, a_L [m^2]


    e0::Float64
    eps0::Float64
    kB::Float64  
    N_A::Float64 
    zA::Float64  
    mO::Float64  
    mZr::Float64 
    mY::Float64
    zL::Float64   # average charge number [1]
    y0::Float64   # electroneutral value [1]
    ML::Float64   # averaged molar mass [kg]
    
    #
    # Ni H2 H2O parameters
    #
     #
    k0::Array{Float64,1}
    vac::Array{Float64,2}
    kfor::Array{Float64,1}
    keqn::Array{Float64,1}
    kinMatrix::Array{Int32,2}    
    kinMatrixActive::Array{Int32,2}    
    kinMatrixReag::Array{Int32,2}    
    kinMatrixProd::Array{Int32,2}    
    num_spec::Int32
    num_reac::Int32
    num_active::Int32
    isgas::Array{Int32,1}
    areaTPB::Float64
    spec_bulk::Array{Float64,1}
    spec_mass::Array{Float64,1}
    spec_charge::Array{Float64,1}

    YSZParameters()= YSZParameters( new())
end

function YSZParameters(this)
    
    this.e0   = 1.602176565e-19  #  [C]
    this.eps0 = 8.85418781762e-12 #  [As/(Vm)]
    this.kB   = 1.3806488e-23  #  [J/K]
    this.N_A  = 6.02214129e23  #  [#/mol]
    this.mO  = 16/1000/this.N_A  #[kg/#]
    this.mZr = 91.22/1000/this.N_A #  [kg/#]
    this.mY  = 88.91/1000/this.N_A #  [kg/#]


    this.A0= 10.0^21.71975544711280
    this.R0= 10.0^20.606423236896422
    this.DGA= 0.0905748 * this.e0 # this.e0 = eV
    this.DGR= -0.708014 * this.e0 
    this.beta= 0.6074566741435283
    this.A= 10.0^0.1
    
   
    #this.DD=1.5658146540360312e-11  # [m / s^2]fitted to conductivity 0.063 S/cm ... TODO reference
    #this.DD=8.5658146540360312e-10  # random value  <<<< GOOOD hand-guess
    this.DD=9.5658146540360312e-10  # some value  <<<< nearly the BEST hand-guess
    #this.DD=9.5658146540360312e-11  # testing value
    this.pO=1.0                   # O2 atmosphere 
    this.T=1073                     
    this.nu=0.9                     # assumption
    this.nus=0.9                    # assumption
    this.x_frac=0.08                # 8% YSZ
    this.chi=27.e0                  # from relative permitivity e_r = 6 = (1 + \chi) ... TODO reference
    this.m_par = 2                  
    this.ms_par = this.m_par        
    this.numax = (2+this.x_frac)/this.m_par/(1+this.x_frac)
    this.nusmax = (2+this.x_frac)/this.ms_par/(1+this.x_frac)
    
    this.vL=3.35e-29 # [m^3]
    this.areaL=(this.vL)^0.6666 # ~ 1.04e-19 [m^2] 
    this.zA  = -2;
    this.zL  = 4*(1-this.x_frac)/(1+this.x_frac) + 3*2*this.x_frac/(1+this.x_frac) - 2*this.m_par*this.nu
    this.y0  = -this.zL/(this.zA*this.m_par*(1-this.nu))
    this.ML  = (1-this.x_frac)/(1+this.x_frac)*this.mZr + 2*this.x_frac/(1+this.x_frac)*this.mY + this.m_par*this.nu*this.mO
    # this.zL=1.8182
    # this.y0=0.9
    # this.ML=1.77e-25    
    #

    # Stoichiometric matrix ()
    this.kinMatrix =  [
    #     φ  O^2- H2 H2O e^-  a   b   c   e   f   g   h   i   j   bv  d    
          0   0  -1   0   0  -1   1   0   0   0   0   0   0   0   0   0; # R1  
          0   0   0   0   1   0  -1   1   0   0   0   0   0   0   0   0; # R2  
          0   0   0   1   1   0   0  -1   0   0   0   0   0   0   0   1; # R3  
          0  -1   0   0   0   1   0   0   0   0   0   0   0   0   1  -1; # R4  
          0   0   0   0   0   0   0  -1   1   0   0   0   0   0   0   0; # R6  
          0  -1   0   0   0   0   0   0  -1   1   0   0   0   0   1   0; # R7  
          0   0   0   1   1   1   0   0   0  -1   0   0   0   0   0   0; # R9  
          0   0   0   0   0   0  -1   0   0   0   1   0   0   0   0   0; # R10 
          0  -1   0   0   0   0   0   0   0   0  -1   1   0   0   1   0; # R11 
          0   0   0   0   1   0   0   0   0   1   0  -1   0   0   0   0; # R13 
          0   0   0   0   0  -1   0   0   0   0   0   0   1   0   0   0; # R14 
          0  -1   0   0   0   0   0   0   0   0   0   0  -1   1   1   0; # R15 
          0   0  -1   0   0   0   0   0   0   0   0   1   0  -1   0   0  # R17 
     ]
    this.keqn =exp.(-(this.e0/this.kB/this.T)*[ #
#      R1,      R2,      R3,      R4,      R6,      R7,      R9,     R10,     R11,     R13,     R14,     R15,   R17
    -1.06;    0.99;    1.19;    0.33;   -0.15;    0.37;    1.21;    0.52;    0.24;    0.42;    0.45;    0.43; -1.21     
     ])
    this.kfor =exp.(-this.e0/this.kB/this.T*[ # 
      0.0;    1.33;    1.28;    0.49;     0.0;    0.41;    1.09;    0.77;    0.47;    1.12;    0.82;    0.44;   0.0
     ])
    this.k0  = [ #
      1.0; 1.69e13; 3.72e13; 1.08e13; 1.00e13; 5.86e12; 3.58e13; 1.10e13; 1.16e13; 4.78e13; 1.07e13; 1.21e13;   1.0
     ]
    this.num_spec=size(this.kinMatrix)[2]
    this.num_active=this.num_spec - 2
    this.num_reac=size(this.kinMatrix)[1]
    this.kinMatrixActive = this.kinMatrix[:, 1:this.num_active]
    this.kinMatrixReag = Int.(1//2*(abs.(this.kinMatrix) + this.kinMatrix))
    this.kinMatrixProd = Int.(1//2*(abs.(this.kinMatrix) - this.kinMatrix))
    #                  φ O^2- H2 H2O e^-   a   b   c   e   f   g   h   i   j   d
    this.isgas      = [0;  0;  1;  1;  1;  0;  0;  0;  0;  0;  0;  0;  0;  0]       # 
    this.spec_bulk  = [0;  1;  1;  1;  1;  0;  0;  0;  0;  0;  0;  0;  0;  0]       #
    this.spec_charge= [0; -2;  0;  0; -1;  0;  0;  1;  1; -1;  0; -2;  0; -2;  -2]   #
    this.areaTPB = 3.7e-19 #3.7e-19 [m^2] Ammal and Heyden 2013
    #                  φ       O^2-       H2       H2O        e^-   a   b   c   e   f   g   h   i   j
    this.spec_mass  = [0; 2.66e-26; 3.35e-27; 2.99e-26; 9.11e-31;   1;  1;  1;  1;  1;  1;  1;  1;  1] # [kg/particle]

    return this
end

function YSZParameters_update(this)
    this.areaL=(this.vL)^0.6666
    this.numax = (2+this.x_frac)/this.m_par/(1+this.x_frac)
    this.nusmax = (2+this.x_frac)/this.ms_par/(1+this.x_frac)   
    
    this.zL  = 4*(1-this.x_frac)/(1+this.x_frac) + 3*2*this.x_frac/(1+this.x_frac) - 2*this.m_par*this.nu
    this.y0  = -this.zL/(this.zA*this.m_par*(1-this.nu))
    this.ML  = (1-this.x_frac)/(1+this.x_frac)*this.mZr + 2*this.x_frac/(1+this.x_frac)*this.mY + this.m_par*this.nu*this.mO
    return this
end


function printfields(this)
    for name in fieldnames(typeof(this))
        @printf("%8s = ",name)
        println(getfield(this,name))
    end
end



#
#
# Ni H2 H2O
#

#
# transient part of Ni measurement functional
#
function currentNi_tran!(meas, u, sys)
    parameters=data(sys)
    U=reshape(u,sys)
    dx_end = sys.grid.coord[1,end] - sys.grid.coord[1,end-1]
    dphi_end = U[1, end] - U[1, end-1]
    dphiB=parameters.eps0*(1+parameters.chi)*(dphi_end/dx_end)
    Qb= - integrate(sys,label_ysz_model.reaction!,U) # \int n^F            
    meas[1] = -Qb[1] - dphiB
end            

#
# steady :) part of Ni measurement functional
#
function currentNi_stdy!(meas, u, sys)
    parameters=data(sys)
    U=reshape(u,sys)
    curr_comb = zeros(parameters.num_reac)
    for ii=1:parameters.num_reac, jj=1:parameters.num_active
        curr_comb[ii] += parameters.kinMatrixActive[ii,jj]*(parameters.spec_charge[jj] - parameters.spec_charge[end]) + parameters.spec_charge[5]*parameters.kinMatrixActive[ii,5]
    end
    reac = reaction_rates(U[:,1], parameters)
    meas[1] = parameters.spec_charge[5]*parameters.e0*sum(curr_comb.*reac )
end

function breactionNi!(f,u,node,data::YSZParameters)
    if  node.region==1
    for ii=3:data.num_active
        f[ii] = (data.spec_mass[ii]*data.spec_bulk[ii])^data.spec_bulk[ii]*(1 - data.isgas[ii])*mass_production(u, data)[ii]
    end
    end
end

function bstorageNi!(f,u,node,data::YSZParameters)
    if  node.region==1
    for ii=3:data.num_active
        f[ii] = data.spec_mass[ii]*u[ii]/data.areaTPB  #[ kg/m^2]
    end
    end
end

function mass_production(u,data)
    RR = reaction_rates(u,data)
    return [# 
            sum(#
               [data.kinMatrixActive[x,y]*RR[x] for x=1:data.num_reac]#
            )/data.areaTPB#
            for y=1:data.num_active#
    ]
end


function reaction_rates(u,data::YSZParameters)
    k0 = data.k0
    yd = [u ; [1- u[2], 1 - sum(u[6:end])]] # [u, [bv, d]]
    kfor = data.kfor
    keqn = data.keqn
    Aprod = data.kinMatrixProd
    Areag = data.kinMatrixReag
    # todo static allocation
    f = Array{eltype(u),1}(undef,data.num_reac)
    for ii=1:data.num_reac
        f[ii] = k0[ii]*kfor[ii]*(prod(yd.^Areag[ii,:]) - keqn[ii]*prod(yd.^Aprod[ii,:]))
    end
    return f
end

end
