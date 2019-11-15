module ysz_model_fitted_parms

using Printf
using VoronoiFVM
using PyPlot
using DataFrames
using CSV
using LeastSquaresOptim


export iphi, iy, ib
const iphi=1
const iy=2
const ib=3


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
    
    this.vL=3.35e-29
    this.areaL=(this.vL)^0.6666
    this.zA  = -2;
    this.zL  = 4*(1-this.x_frac)/(1+this.x_frac) + 3*2*this.x_frac/(1+this.x_frac) - 2*this.m_par*this.nu
    this.y0  = -this.zL/(this.zA*this.m_par*(1-this.nu))
    this.ML  = (1-this.x_frac)/(1+this.x_frac)*this.mZr + 2*this.x_frac/(1+this.x_frac)*this.mY + this.m_par*this.nu*this.mO
    # this.zL=1.8182
    # this.y0=0.9
    # this.ML=1.77e-25    
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




# time derivatives
function storage!(f,u, node, this::YSZParameters)
    f[iphi]=0
    f[iy]=this.mO*this.m_par*(1.0-this.nu)*u[iy]/this.vL
end

function bstorage!(f,u,node, this::YSZParameters)
    if  node.region==1
        f[ib]=this.mO*this.ms_par*(1.0-this.nus)*u[ib]/this.areaL
    else
        f[ib]=0
    end
end

# bulk flux
function flux!(f,u, edge, this::YSZParameters)
    uk=viewK(edge,u)
    ul=viewL(edge,u)
    f[iphi]=this.eps0*(1+this.chi)*(uk[iphi]-ul[iphi])    
    
    bp,bm=fbernoulli_pm(
        (1.0 + this.mO/this.ML*this.m_par*(1.0-this.nu))
        *(log(1-ul[iy]) - log(1-uk[iy]))
        -
        this.zA*this.e0/this.T/this.kB*(
            1.0 + this.mO/this.ML*this.m_par*(1.0-this.nu)*0.5*(uk[iy]+ul[iy])
        )*(ul[iphi] - uk[iphi])
    )
    f[iy]= (
        this.DD
        *
        (1.0 + this.mO/this.ML*this.m_par*(1.0-this.nu)*0.5*(uk[iy]+ul[iy]))
        *
        this.mO*this.m_par*(1.0-this.nu)/this.vL
        *
        (bm*uk[iy]-bp*ul[iy])
    )
end


# sources
function reaction!(f,u, node, this::YSZParameters)
    f[iphi]=-(this.e0/this.vL)*(this.zA*this.m_par*(1-this.nu)*u[iy] + this.zL) # source term for the Poisson equation, beware of the sign
    f[iy]=0
end

# surface reaction
function electroreaction(this::YSZParameters, u)
    if this.R0 > 0
        eR = (
            this.R0
            *(
                exp(-this.beta*this.A*this.DGR/(this.kB*this.T))
                *(u/(1-u))^(-this.beta*this.A)
                *(this.pO)^(this.beta*this.A/2.0)
                - 
                exp((1.0-this.beta)*this.A*this.DGR/(this.kB*this.T))
                *(u/(1-u))^((1.0-this.beta)*this.A)
                *(this.pO)^(-(1.0-this.beta)*this.A/2.0)
            )
        )
    else
        eR=0
    end
end

# surface reaction + adsorption
function breaction!(f,u,node,this::YSZParameters)
    if  node.region==1
        electroR=electroreaction(this,u[ib])
        f[iy]= (
            this.mO*this.A0*
            (
                - this.DGA/(this.kB*this.T) 
                +    
                log(u[iy]*(1-u[ib]))
                - 
                log(u[ib]*(1-u[iy]))
            )
        )
        # if bulk chem. pot. > surf. ch.p. then positive flux from bulk to surf
        # sign is negative bcs of the equation implementation
        f[ib]= (
            - this.mO*electroR - this.mO*this.A0*
            (
                - this.DGA/(this.kB*this.T) 
                +    
                log(u[iy]*(1-u[ib]))
                - 
                log(u[ib]*(1-u[iy]))
                
            )
        )      
        f[iphi]=0
    else
        f[iy]=0
        f[iphi]=0
    end
end



function breaction2!(f,u,node,this::YSZParameters)
  if  node.region==1
      f[iy]=(u[iy]-u[ib])
      f[ib]=(u[ib]-u[iy])
  else
      f[1]=0
      f[2]=0
  end
end

function direct_capacitance(this::YSZParameters, domain)
    # Clemens' analytic solution
    #printfields(this)
    
    PHI = domain
    #PHI=collect(-1:0.01:1) # PHI = phi_B-phi_S, so domain as phi_S goes with minus
    my_eps = 0.001
    for i in collect(1:length(PHI))
        if abs(PHI[i]) < my_eps
            PHI[i]=my_eps
        end
    end
    #
    #yB = -this.zL/this.zA/this.m_par/(1-this.nu);
    yB = this.y0
    X= yB/(1-yB)*exp.(.- this.zA*this.e0/this.kB/this.T*PHI)
    y  = X./(1.0.+X)
    #
    nF = this.e0/this.vL*(this.zL .+ this.zA*this.m_par*(1-this.nu)*y)

    
    F  = - sign.(PHI).*sqrt.(
          2*this.e0/this.vL/this.eps0/(1.0+this.chi).*(
            .- this.zL.*PHI .+ this.kB*this.T/this.e0*this.m_par*(1-this.nu)*log.(
              (1-yB).*(X .+ 1.0)
             )
           )
         );
    #
    Y  = yB/(1-yB)*exp.(- this.DGA/this.kB/this.T .- this.zA*this.e0/this.kB/this.T*PHI);
    #
    CS = this.zA^2*this.e0^2/this.kB/this.T*this.ms_par/this.areaL*(1-this.nus)*Y./((1.0.+Y).^2);
    CBL  = nF./F;
    return CBL, CS, y
end

# conversions for the equilibrium case
function y0_to_phi(this::YSZParameters, y0)
    yB = -this.zL/this.zA/this.m_par/(1-this.nu);
    return - (this.kB * this.T / (this.zA * this.e0)) * log(y0/(1-y0) * (1-yB)/yB )
end

function phi_to_y0(this::YSZParameters, phi)
    yB = -this.zL/this.zA/this.m_par/(1-this.nu);
    X  = yB/(1-yB)*exp.(this.zA*this.e0/this.kB/this.T* (-phi))
    return X./(1.0.+X)
end

function equil_phi(this::YSZParameters)
    B = exp( - (- this.DGA + this.DGR) / (this.kB * this.T))*this.pO^(1/2.0)
    return y0_to_phi(this, B/(1+B))
end


end
