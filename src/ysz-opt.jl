    """
YSZ example cloned from iliq.jl of the TwoPointFluxFVM package.
"""

using Printf
using TwoPointFluxFVM
using PyPlot
using DataFrames
using CSV
using LeastSquaresOptim

mutable struct YSZParameters <:TwoPointFluxFVM.Physics
    TwoPointFluxFVM.@AddPhysicsBaseClassFields

    # to fit
    A0::Float64   # surface adsorption coefficient [ s^-1 ]
    R0::Float64 # exhange current density [m^-2 s^-1]
    DGA::Float64 # difference of gibbs free energy of adsorbtion
    DGR::Float64 # difference of gibbs free energy of electrochemical reaction
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
    TwoPointFluxFVM.PhysicsBase(this,2)
    this.num_bspecies=[ 1, 0]
    
    
    this.e0   = 1.602176565e-19  #  [C]
    this.eps0 = 8.85418781762e-12 #  [As/(Vm)]
    this.kB   = 1.3806488e-23  #  [J/K]
    this.N_A  = 6.02214129e23  #  [#/mol]
    this.mO  = 16/1000/this.N_A  #[kg/#]
    this.mZr = 91.22/1000/this.N_A #  [kg/#]
    this.mY  = 88.91/1000/this.N_A #  [kg/#]


    this.A0=1e-1#.e3
    this.R0=1.0e-2
    this.DGA= this.e0 # equivalent to 1eV
    this.DGR=0.0e-2
    this.beta=0.5
    this.A=2.
    
    
    #this.DD=1.5658146540360312e-11  # fitted to conductivity 0.063 S/cm ... TODO reference
    #this.DD=8.5658146540360312e-10  # random value  <<<< GOOOD hand-guess
    this.DD=9.5658146540360312e-10  # some value  <<<< nearly the BEST hand-guess
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


const iphi=1
const iy=2

# time derivatives
function storage!(this::YSZParameters, f,u)
    f[iphi]=0
    f[iy]=this.mO*this.m_par*(1.0-this.nu)*u[iy]/this.vL
end

function bstorage!(this::YSZParameters,bf,bu)
    if  this.bregion==1
        bf[1]=this.mO*this.ms_par*(1.0-this.nus)*bu[1]/this.areaL
    else
        bf[1]=0
    end
end

# bulk flux
function flux!(this::YSZParameters,f,uk,ul)
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
function reaction!(this::YSZParameters, f,u)
    f[iphi]=-(this.e0/this.vL)*(this.zA*this.m_par*(1-this.nu)*u[iy] + this.zL) # source term for the Poisson equation, beware of the sign
    f[iy]=0
end

# surface reaction
function electroreaction(this::YSZParameters, bu)
    if this.R0 > 0
        eR = (
            this.R0
            *(
                exp(-this.beta*this.A*this.DGR/(this.kB*this.T))
                *(bu[1]/(1-bu[1]))^(-this.beta*this.A)
                *(this.pO)^(this.beta*this.A/2.0)
                - 
                exp((1.0-this.beta)*this.A*this.DGR/(this.kB*this.T))
                *(bu[1]/(1-bu[1]))^((1.0-this.beta)*this.A)
                *(this.pO)^(-(1.0-this.beta)*this.A/2.0)
            )
        )
    else
        eR=0
    end
end

# surface reaction + adsorption
function breaction!(this::YSZParameters,f,bf,u,bu)
    if  this.bregion==1
        electroR=electroreaction(this,bu)
        f[iy]= (
            this.A0*(this.mO/this.areaL)*
            (
                - this.DGA/(this.kB*this.T) 
                +    
                log(u[iy]*(1-bu[1]))
                - 
                log(bu[1]*(1-u[iy]))
            )
        )
        # if bulk chem. pot. > surf. ch.p. then positive flux from bulk to surf
        # sign is negative bcs of the equation implementation
        bf[1]= (
            - this.mO*electroR - this.A0*(this.mO/this.areaL)*
            (
                - this.DGA/(this.kB*this.T) 
                +    
                log(u[iy]*(1-bu[1]))
                - 
                log(bu[1]*(1-u[iy]))
                
            )
        )      
        f[iphi]=0
    else
        f[iy]=0
        f[iphi]=0
    end
end



function breaction2!(this::YSZParameters,f,bf,u,bu)
  if  this.bregion==1
      f[iy]=(u[iy]-bu[1])
      bf[1]=(bu[1]-u[iy])
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


###########################################################
### Supporting functions ##################################
function get_unknowns(x, X, U) # at point x
    xk = 0
    for i in collect(1:length(X)-1)
#        println("i ",i)
#        println("x / xk / xl   ", x, " / ", X[i]," / ", X[i + 1])
        if (X[i] <= x) && (x <= X[i+1])
            xk = i
            break
        end
    end
    if xk == 0 
        println("ERROR: get_uknowns: x ",x," is not in X = [",X[1],", ",X[end],"]")
        #return [0, 0]
    end
#    println("x / xk / xl   ", x, " / ", xk," / ", xk + 1)
#    println("length of U    ", length(U))
    Uk=U[:,xk]
    Ul=U[:,xk+1]
    return Uk + ((Ul - Uk)/(X[xk+1] - X[xk])) * (x - X[xk])
end

function get_material_flux(this::YSZParameters, uk, ul, h)
    y = 0.5*(uk[iy]+ul[iy])
    return (
        -this.DD
        *            
        this.mO*this.m_par*(1.0-this.nu)/this.vL
        *
        (1.0 + this.mO/this.ML*this.m_par*(1.0-this.nu)*y)^2
        *
        (
            (ul[iy] - uk[iy])/(h*(1-y))
            +
            this.zA*y*(this.e0/(this.kB*this.T))*(ul[iphi] - uk[iphi])/h
        )
    )
end

function get_DGR_electroneutral(this::YSZParameters)
    return this.kB*this.T*log(
        (-this.zA*this.m_par*(1-this.nu) + this.zL)
        /
        (this.zL)
        *
        this.pO^(0.5)
    )
end

###########################################################
###########################################################
###########################################################
### Fitting stuff #########################################

function is_between(x, a, b)
    if (b - a) > 0
        if (a <= x) & (x <= b)
            return true
        end
    else
        if (b <= x) & (x <= a)
            return true
        end
    end
    return false
end

function check_nodes_short()
    return [
        [0.01, 1, 0.015686332167948364],
        [0.1, 1, 0.046800590942188185],
        [0.4, 1, 0.37951877565853265],
        [0.2, -1, 0.18360864356578493],
        [-0.2, -1, -0.18914394378285262],
        [-0.4, 1, -0.758257259313434],
        [-0.1, 1, -0.04691697173124805]
        ]
end

function check_nodes_long()
    return [
    [ 0.01 ,  1 ,  0.0156863321679 ],
    [ 0.05 ,  1 ,  0.0281101995946 ],
    [ 0.1 ,  1 ,  0.0468005909422 ],
    [ 0.15 ,  1 ,  0.0716042511468 ],
    [ 0.2 ,  1 ,  0.105453784583 ],
    [ 0.25 ,  1 ,  0.150762565352 ],
    [ 0.3 ,  1 ,  0.210008281387 ],
    [ 0.35 ,  1 ,  0.285365362973 ],
    [ 0.4 ,  1 ,  0.379518775659 ],
    [ 0.4 ,  -1 ,  0.432316804894 ],
    [ 0.35 ,  -1 ,  0.354766238866 ],
    [ 0.3 ,  -1 ,  0.286613797492 ],
    [ 0.25 ,  -1 ,  0.231818661618 ],
    [ 0.2 ,  -1 ,  0.183608643566 ],
    [ 0.15 ,  -1 ,  0.13934314172 ],
    [ 0.1 ,  -1 ,  0.0974983134916 ],
    [ 0.05 ,  -1 ,  0.0580951170398 ],
    [ 0.0 ,  -1 ,  0.0178260739169 ],
    [ -0.05 ,  -1 ,  -0.0284506730363 ],
    [ -0.1 ,  -1 ,  -0.0767715103415 ],
    [ -0.15 ,  -1 ,  -0.126002461059 ],
    [ -0.2 ,  -1 ,  -0.189143943783 ],
    [ -0.25 ,  -1 ,  -0.279117596195 ],
    [ -0.3 ,  -1 ,  -0.403615421853 ],
    [ -0.35 ,  -1 ,  -0.557567899648 ],
    [ -0.4 ,  -1 ,  -0.741932592896 ],
    [ -0.4 ,  1 ,  -0.758257259313 ],
    [ -0.35 ,  1 ,  -0.567389007323 ],
    [ -0.3 ,  1 ,  -0.406031747523 ],
    [ -0.25 ,  1 ,  -0.265960837541 ],
    [ -0.2 ,  1 ,  -0.168469540759 ],
    [ -0.15 ,  1 ,  -0.101001232934 ],
    [ -0.1 ,  1 ,  -0.0469169717312 ],
    [ -0.05 ,  1 ,  -0.00825189012041 ],
    [ -0.01 ,  1 ,  0.00942419878596 ],
    ]
end

function check_nodes_whole_IRcorr()
    return [
    [ 0.01 ,  1 ,  0.0150491542661 ],
    [ 0.05 ,  1 ,  0.0268974898822 ],
    [ 0.1 ,  1 ,  0.0442500369024 ],
    [ 0.15 ,  1 ,  0.0664720034828 ],
    [ 0.2 ,  1 ,  0.0956120904493 ],
    [ 0.25 ,  1 ,  0.132673892311 ],
    [ 0.3 ,  1 ,  0.179171966986 ],
    [ 0.35 ,  1 ,  0.234708250962 ],
    [ 0.4 ,  1 ,  0.300126956062 ],
    [ 0.473 ,  1 ,  0.413006128957 ],
    [ 0.473 ,  -1 ,  0.45061070576 ],
    [ 0.4 ,  -1 ,  0.356226587042 ],
    [ 0.35 ,  -1 ,  0.297462116657 ],
    [ 0.3 ,  -1 ,  0.248312046157 ],
    [ 0.25 ,  -1 ,  0.204239402861 ],
    [ 0.2 ,  -1 ,  0.163227207933 ],
    [ 0.15 ,  -1 ,  0.124400778752 ],
    [ 0.1 ,  -1 ,  0.0878372675004 ],
    [ 0.05 ,  -1 ,  0.052540481894 ],
    [ 0.0 ,  -1 ,  0.0159355606336 ],
    [ -0.05 ,  -1 ,  -0.0251917337437 ],
    [ -0.1 ,  -1 ,  -0.0680737763948 ],
    [ -0.15 ,  -1 ,  -0.110758965865 ],
    [ -0.2 ,  -1 ,  -0.159671398554 ],
    [ -0.25 ,  -1 ,  -0.22065905286 ],
    [ -0.3 ,  -1 ,  -0.29857114029 ],
    [ -0.35 ,  -1 ,  -0.392319818746 ],
    [ -0.4 ,  -1 ,  -0.498452273921 ],
    [ -0.428 ,  -1 ,  -0.560535107789 ],
    [ -0.428 ,  1 ,  -0.566981785728 ],
    [ -0.4 ,  1 ,  -0.507237810513 ],
    [ -0.35 ,  1 ,  -0.393025970981 ],
    [ -0.3 ,  1 ,  -0.290811538942 ],
    [ -0.25 ,  1 ,  -0.205435404638 ],
    [ -0.2 ,  1 ,  -0.140370670431 ],
    [ -0.15 ,  1 ,  -0.0869515426903 ],
    [ -0.1 ,  1 ,  -0.0415431908515 ],
    [ -0.05 ,  1 ,  -0.00766406292301 ],
    [ -0.01 ,  1 ,  0.00898350750788 ],
    ]
end

function check_nodes_whole()
    return [
    [ 0.01 ,  1 ,  0.0150491542661 ],
    [ 0.05 ,  1 ,  0.0268974898822 ],
    [ 0.1 ,  1 ,  0.0442500369024 ],
    [ 0.15 ,  1 ,  0.0664720034828 ],
    [ 0.2 ,  1 ,  0.0956120904493 ],
    [ 0.25 ,  1 ,  0.132673892311 ],
    [ 0.3 ,  1 ,  0.179171966986 ],
    [ 0.35 ,  1 ,  0.234708250962 ],
    [ 0.4 ,  1 ,  0.300126956062 ],
    [ 0.45 ,  1 ,  0.375883444117 ],
    [ 0.5 ,  1 ,  0.458309824876 ],
    [ 0.55 ,  1 ,  0.548480431896 ],
    [ 0.55 ,  -1 ,  0.548925634441 ],
    [ 0.5 ,  -1 ,  0.488403386928 ],
    [ 0.45 ,  -1 ,  0.41946584191 ],
    [ 0.4 ,  -1 ,  0.356226587042 ],
    [ 0.35 ,  -1 ,  0.297462116657 ],
    [ 0.3 ,  -1 ,  0.248312046157 ],
    [ 0.25 ,  -1 ,  0.204239402861 ],
    [ 0.2 ,  -1 ,  0.163227207933 ],
    [ 0.15 ,  -1 ,  0.124400778752 ],
    [ 0.1 ,  -1 ,  0.0878372675004 ],
    [ 0.05 ,  -1 ,  0.052540481894 ],
    [ 0.0 ,  -1 ,  0.0159355606336 ],
    [ -0.05 ,  -1 ,  -0.0251917337437 ],
    [ -0.1 ,  -1 ,  -0.0680737763948 ],
    [ -0.15 ,  -1 ,  -0.110758965865 ],
    [ -0.2 ,  -1 ,  -0.159671398554 ],
    [ -0.25 ,  -1 ,  -0.22065905286 ],
    [ -0.3 ,  -1 ,  -0.29857114029 ],
    [ -0.35 ,  -1 ,  -0.392319818746 ],
    [ -0.4 ,  -1 ,  -0.498452273921 ],
    [ -0.45 ,  -1 ,  -0.6116602738 ],
    [ -0.5 ,  -1 ,  -0.736105820118 ],
    [ -0.547 ,  -1 ,  -0.857567478585 ],
    [ -0.547 ,  1 ,  -0.858335657811 ],
    [ -0.5 ,  1 ,  -0.746960182353 ],
    [ -0.45 ,  1 ,  -0.620437414239 ],
    [ -0.4 ,  1 ,  -0.507237810513 ],
    [ -0.35 ,  1 ,  -0.393025970981 ],
    [ -0.3 ,  1 ,  -0.290811538942 ],
    [ -0.25 ,  1 ,  -0.205435404638 ],
    [ -0.2 ,  1 ,  -0.140370670431 ],
    [ -0.15 ,  1 ,  -0.0869515426903 ],
    [ -0.1 ,  1 ,  -0.0415431908515 ],
    [ -0.05 ,  1 ,  -0.00766406292301 ],
    [ -0.01 ,  1 ,  0.00898350750788 ],
    ]
end

function CV_get_error(CV_U, CV_I, nodes; opt_pyplot=false)
        #res_values = []
        Ux_values = []
        err = 0
        #println(CV_U)
        #println(CV_I)

        start_i = 1
        for node in nodes
            #print nodes
            direction = 1
            Ux = node[1]
            xk = -1
            i = start_i
            while i < length(CV_U)-1
                if direction*(CV_U[i+1] - CV_U[i]) < 0
                    #print "                          TED "
                    direction *= -1
                end
                if is_between(Ux, CV_U[i], CV_U[i+1]) & (direction == node[2]) 
                    xk = i
                    start_i = i
                    break
                end
                i += 1
            end
            
            if xk == -1
                print("Error: CV_get_values: Ux ",Ux," is not in CV_U")
                #res_values.append(0.0)
                continue
            end
            Ik = CV_I[xk]
            Il = CV_I[xk+1]
            Ix = Ik + ((Il - Ik)/(CV_U[xk+1] - CV_U[xk])) * (Ux - CV_U[xk])
            
            #res_values.append(float(Ix))        
            err += (node[3] - float(Ix))^2
        end

        U_orig = zeros(0)
        I_orig = zeros(0)
        for i in nodes
            append!(U_orig,i[1])
            append!(I_orig,i[3])
        end
        
        if opt_pyplot
            PyPlot.close
            #PyPlot.clf()
            #subplot(211)
            #PyPlot.figure(figsize=(5.3, 4))
            PyPlot.plot(CV_U,CV_I, label="sim")

            PyPlot.plot(U_orig,I_orig,label="exp")
            PyPlot.grid()
            PyPlot.xlabel("nu (V)")
            PyPlot.ylabel("I (A)")
            PyPlot.legend(loc="best")
            
            #subplot(212)
            #PyPlot.plot(CV_U,CV_I)
            #PyPlot.grid()
            
            PyPlot.draw()
            PyPlot.show()
            PyPlot.pause(1)
        end
        
        return sqrt(err) / convert(Float64, length(nodes))
    end

###########################################################
###########################################################

function run_new(;hexp=-9, verbose=false ,pyplot=false, width=10.0e-9, voltametry=false, dlcap=false, save_files=false, voltrate=0.005, phi0=0.0, upp_bound=0.5, low_bound=-0.5, sample=50, prms_in=[-10, 10, 0.0, 0.1, 0.5, 0.1], dtstep_in=1.0e-6, fitting=false, print_bool=false, par_study_bool=false, nu_in=0.9 )

    # A0_in \in [-6, 6]
    # R0_in \in [-6, 6]
    # DGA_in \in [0, 1]
    # DGR_in \in [-2, 2]
    # beta_in \in [0,1]
    # A_in \in [-1, 1]
    #
    # AREA OF THE ELECTROLYTE CUT
    AreaEllyt = 0.000201 * 0.6      # m^2     (1 - porosity)
    width_Ellyt = 0.00045           # m     width of the half-cell
    if dlcap
        AreaEllyt = 1.0      # m^2     (1 - porosity)
        println("dlcap > area = 1")
    end
    #
    dx_start = 10^convert(Float64,hexp)
    

    if true
        # only double layer
        X=width*TwoPointFluxFVM.geomspace(0.0,1.0,dx_start,1e-1)
    else
        # the whole half-cell
        # does not work well because of poor integration over domain ... IMHO
        DL = width*TwoPointFluxFVM.geomspace(0.0,1.0,dx_start,1e-1)
        HalfBULK = width_Ellyt*TwoPointFluxFVM.geomspace(width/width_Ellyt,1.0,1.0e-10/width_Ellyt,(1.0e-5)/width_Ellyt)
        deleteat!(HalfBULK,1)
        X = vcat(DL,HalfBULK)
    end
    #println("X = ",X)
    
    #
    geom=TwoPointFluxFVM.Graph(X)
    #
    
    parameters=YSZParameters()
    # for parametric study
    eV = parameters.e0   # electronvolt [J] = charge of electron * 1[V]
    parameters.A0 = 10.0^prms_in[1]      # [1 / s]
    parameters.R0 = 10.0^prms_in[2]      # [1 / m^2 s]
    if dlcap
        parameters.R0 = 0
        println("dlcap > R0= ",parameters.R0)
    end
    parameters.DGA = prms_in[3] * eV    # [J]
    parameters.DGR = prms_in[4] * eV    # [J]
    
    #parameters.DGR = get_DGR_electroneutral(parameters)
    
    parameters.beta = prms_in[5]       # [1]
    parameters.A = 10.0^prms_in[6]        # [1]
    
    
    parameters.nu = nu_in
    parameters = YSZParameters_update(parameters)
    
    #
    parameters.storage=storage!
    parameters.flux=flux!
    parameters.reaction=reaction!
    parameters.breaction=breaction!
    parameters.bstorage=bstorage!
    #
    if print_bool
        printfields(parameters)
    end
    #print("weight ", parameters.mO*parameters.m_par*(1.0-parameters.nu)/parameters.vL,"\n")
    #
    sys=TwoPointFluxFVM.System(geom,parameters)
    #
    #sys.boundary_values[iphi,1]=1.0e-0
    sys.boundary_values[iphi,2]=0.0e-3
    #
    sys.boundary_factors[iphi,1]=TwoPointFluxFVM.Dirichlet
    sys.boundary_factors[iphi,2]=TwoPointFluxFVM.Dirichlet
    #
    sys.boundary_values[iy,2]=parameters.y0
    sys.boundary_factors[iy,2]=TwoPointFluxFVM.Dirichlet
    #
    inival=unknowns(sys)
    inival.=0.0
    #
    inival_bulk=bulk_unknowns(sys,inival)
    
    phi0 = equil_phi(parameters)
    if print_bool
        println("phi0 = ",phi0)
    end
    if dlcap
        phi0 = 0
        if print_bool
            println("dlcap > phi0= ", phi0)
        end
    end


    for inode=1:size(inival_bulk,2)
        #inival_bulk[iphi,inode]=0.0e-3
        inival_bulk[iy,inode]= parameters.y0
    end
    inival_boundary = boundary_unknowns(sys,inival,1)
    inival_boundary[1]= parameters.y0
    #
    control=TwoPointFluxFVM.NewtonControl()
    control.verbose=verbose
    control.tol_linear=1.0e-4
    control.tol_relative=1.0e-5
    #control.tol_absolute=1.0e-4
    #control.max_iterations=3
    control.max_lureuse=0
    control.damp_initial=1.0e-5
    control.damp_growth=1.9
    time=0.0
    if (!voltametry)
        println("---------- testsing branch ------------")
        
        sys=TwoPointFluxFVM.System(geom,parameters)
        
        sys.boundary_values[iphi,1]=1.0e-8
        sys.boundary_values[iphi,2]=0.0e-0
        #
        sys.boundary_factors[iphi,1]=TwoPointFluxFVM.Dirichlet
        sys.boundary_factors[iphi,2]=TwoPointFluxFVM.Dirichlet
        #
        sys.boundary_values[iy,1]=parameters.y0
        sys.boundary_values[iy,2]=parameters.y0
        #
        sys.boundary_factors[iy,1]=TwoPointFluxFVM.Dirichlet
        sys.boundary_factors[iy,2]=TwoPointFluxFVM.Dirichlet
        
        # initialization
        inival=unknowns(sys)
        inival.=0.0
        #
        inival_bulk=bulk_unknowns(sys,inival)
        for inode=1:size(inival_bulk,2)
            #inival_bulk[iphi,inode]=0.0e-3
            inival_bulk[iy,inode]= parameters.y0
        end
        inival_boundary = boundary_unknowns(sys,inival,1)
        inival_boundary[1]= parameters.y0
        
        
        t_end = 100
        tstep = 1.0e+1
        t = 0
        while t < t_end
            if print_bool
                println("t = ",t)
            end
            U=solve(sys,inival,control=control,tstep=tstep)
            U_bulk=bulk_unknowns(sys,U)
            U_bound=boundary_unknowns(sys,U,1)
            inival.=U
            
            h = 1.0e-10
            xk = 1.0e-9
            xl = xk + h
            
            Uxk = get_unknowns(xk,X,U_bulk)
            Uxl = get_unknowns(xl,X,U_bulk)
            
            
            #println("Uxk ",Uxk)
            #println("Uxl ",Uxl)
            #println("diff ",Uxl - Uxk)
            #println("flux ",get_material_flux(parameters,Uxk,Uxl,h))
            
            
            
            
            if pyplot
                PyPlot.clf()
                subplot(211)
                plot((10^9)*X[:],U_bulk[1,:],label="phi (V)")
                plot((10^9)*X[:],U_bulk[2,:],label="y")
                PyPlot.xlim(-0.000000001,10)
                PyPlot.xlabel("x (nm)")
                PyPlot.legend(loc="best")
                PyPlot.grid()
                
                subplot(212)
                plot((10^9)*X[:],U_bulk[1,:],label="phi (V)")
                
                pause(0.1)
            end    
            t = t + tstep
            
        end
        #PyPlot.close()
                    
    #
    # voltametry
    #
    end
    if voltametry
        istep=1
        phi=0
#        phi=phi0       
        U0_range=zeros(0)
        Ub_range=zeros(0)
        phi_range=zeros(0)
        phi_range_full=zeros(0)
        Is_range=zeros(0)
        Ib_range=zeros(0)
        Ibb_range=zeros(0)
        r_range=zeros(0)
        
        #MY_Iflux_range=zeros(0)
        MY_Iflux0_range = zeros(0)
        MY_Itot_range=zeros(0)
        if save_files
            out_df = DataFrame(t = Float64[], U = Float64[], Itot = Float64[], Ibu = Float64[], Isu = Float64[], Ire = Float64[])
        end
        
        cv_cycles = 1
        relaxation_length = 1    # how many "samples" should relaxation last
        relax_counter = 0
        istep_cv_start = -1
        time_range = zeros(0)  # [s]

        if print_bool
            print("calculating linear potential sweep\n")
        end
        direction_switch = 0
        
        
        
        
        dtstep=dtstep_in
        
        
        #phi0 = 0
        
        
        
        tstep=((upp_bound-low_bound)/2)/voltrate/sample   
        if print_bool
            @printf("tstep %g ... dtstep %g\n",tstep, dtstep)
        end
        if dtstep > tstep
            dtstep=0.05*tstep
            if print_bool
                print("dtstep refinement: ")
            end
        end
        if phi0 > 0
            dir=1
        else
            dir=-1
        end
        
        if pyplot
            PyPlot.close()
            PyPlot.ion()
            #PyPlot.figure(figsize=(10,8), )
            PyPlot.figure(figsize=(5,5))
        end
        
        state = "ramp"
        if print_bool
            println("phi_equilibrium = ",phi0)
            println("ramp ......")
        end    
        
        
        while state != "cv_is_off"
            if state=="ramp" && ((dir==1 && phi > phi0) || (dir==-1 && phi < phi0))
                phi = phi0
                state = "relaxation"
                if print_bool
                    println("relaxation ... ")
                end
            end            
            if state=="relaxation" && relax_counter == sample*relaxation_length
                relax_counter += 1
                state="cv_is_on"
                istep_cv_start = istep
                dir=1
                if print_bool
                    print("cv ~~~ direction switch: ")
                end
            end                            
            if state=="cv_is_on" && (phi <= low_bound-0.00000001+phi0 || phi >= upp_bound+0.00000001+phi0)
                dir*=(-1)
                # uncomment next line if phi should NOT go slightly beyond limits
                #phi+=2*voltrate*dir*tstep
            
                direction_switch +=1
                if print_bool
                    print(direction_switch,", ")
                end
            end            
            if state=="cv_is_on" && (dir > 0) && (phi > phi0 + 0.000001) && (direction_switch >=2*cv_cycles)
                state = "cv_is_off"
            end
            
            
            # tstep to potential phi
            sys.boundary_values[iphi,1]=phi
            U=solve(sys,inival,control=control,tstep=tstep)
            
            
            Qb= - integrate(sys,reaction!,U) # \int n^F
            
            dphi_end = bulk_unknowns(sys,U)[iphi, end] - bulk_unknowns(sys,U)[iphi, end-1]
            dx_end = X[end] - X[end-1]
            dphiB=parameters.eps0*(1+parameters.chi)*(dphi_end/dx_end)
            
            y_bound=boundary_unknowns(sys,U,1)
            Qs= (parameters.e0/parameters.areaL)*parameters.zA*y_bound*parameters.ms_par*(1-parameters.nus) # (e0*zA*nA_s)

                 
             # for faster computation, solving of "dtstep problem" is not performed
            U0 = inival
            inival.=U
            Qb0 = - integrate(sys,reaction!,U0) # \int n^F
            dphi0_end = bulk_unknowns(sys,U0)[iphi, end] - bulk_unknowns(sys,U0)[iphi, end-1]
            dphiB0 = parameters.eps0*(1+parameters.chi)*(dphi0_end/dx_end)
            y0_bound=boundary_unknowns(sys,U0,1)
            Qs0 = (parameters.e0/parameters.areaL)*parameters.zA*y0_bound*parameters.ms_par*(1-parameters.nus) # (e0*zA*nA_s)


            
            # time derivatives
            Is  = - (Qs[1] - Qs0[1])/tstep                
            Ib  = - (Qb[iphi] - Qb0[iphi])/tstep 
            Ibb = - (dphiB - dphiB0)/tstep
            
            
            # reaction average
            reac = - 2*parameters.e0*electroreaction(parameters, y_bound)
            reacd = - 2*parameters.e0*electroreaction(parameters, y0_bound)
            Ir= 0.5*(reac + reacd)

            #############################################################
            #multiplication by area of electrode I = A * ( ... )
            Ibb = Ibb*AreaEllyt
            Ib = Ib*AreaEllyt
            Is = Is*AreaEllyt
            Ir = Ir*AreaEllyt
            #
            
            
            #@printf("t = %g     U = %g   state = %s  reac = %g  \n", istep*tstep, phi, state, Ir)
            
            if verbose
                #debug(parameters,U, U_bound)
            
            end
            U_bulk=bulk_unknowns(sys,U)
            U_bound=boundary_unknowns(sys,U,1)
            
            
            # control current by flux
            #h = 1.0e-10
            #xk = 5.0e-9
            #xl = xk + h
            #Uxk = get_unknowns(xk,X,U_bulk)
            #Uxl = get_unknowns(xl,X,U_bulk)
            
            #@printf("t = %g     U = %g   state = %s  reac = %g  \n", istep*tstep, phi, state, Ir)
            #println("Uxk ",Uxk)
            #println("Uxl ",Uxl)
            #println("diff ",Uxl - Uxk)
            #j_om = get_material_flux(parameters,Uxk,Uxl,h)
            #I_flux = AreaEllyt*j_om*(-2)*parameters.e0/parameters.mO
            #println("flux j_om ",j_om, "    current ", I_flux, "  I ",Ibb+Is+Ib+Ir)
            
            
            # storing data
            append!(time_range,tstep*istep)
            
            append!(U0_range,U_bulk[iy,1])
            append!(Ub_range,U_bound[1,1])
            append!(phi_range_full,phi)

            append!(phi_range,phi)
            append!(Ib_range,Ib)
            append!(Is_range,Is)
            append!(Ibb_range,Ibb)
            append!(r_range, Ir)
            
            #append!(MY_Iflux_range,I_flux)
            append!(MY_Itot_range,Ibb+Is+Ib+Ir)            
               
            if save_files
                if state=="cv_is_on"
                    if dlcap
                        push!(out_df,[(istep-istep_cv_start)*tstep   phi-phi0    (Ib+Is+Ir)/voltrate    Ib/voltrate    Is/voltrate    Ir/voltrate])
                    else
                        push!(out_df,[(istep-istep_cv_start)*tstep   phi-phi0    Ib+Is+Ir    Ib    Is    Ir])
                    end
                end
            end
            
            
            
            ##### my plotting                  
            num_subplots=2
            ys_marker_size=4
            PyPlot.subplots_adjust(hspace=0.3)
            
            if pyplot && istep%10 == 0
                
                PyPlot.clf() 
                
                
                if num_subplots > 0
                    subplot(num_subplots*100 + 11)
                    plot((10^9)*X[:],U_bulk[iphi,:],label="phi (V)")
                    plot((10^9)*X[:],U_bulk[iy,:],label="y")
                    plot(0,U_bound[1,1],"go", markersize=ys_marker_size, label="y_s")
                    l_plot = 5.0
                    PyPlot.xlim(-0.01*l_plot, l_plot)
                    PyPlot.ylim(-0.5,1.1)
                    PyPlot.xlabel("x (nm)")
                    PyPlot.legend(loc="best")
                    PyPlot.grid()
                end
                
                if num_subplots > 1
                    subplot(num_subplots*100 + 12)
                    plot((10^3)*X[:],U_bulk[iphi,:],label="phi (V)")
                    plot((10^3)*X[:],U_bulk[iy,:],label="y")
                    plot(0,U_bound[1,1],"go", markersize=ys_marker_size, label="y_s")
                    l_plot = 5.0
                    #PyPlot.xlim(-0.01*l_plot, l_plot)
                    PyPlot.ylim(-0.5,1.1)
                    PyPlot.xlabel("x (mm)")
                    PyPlot.legend(loc="best")
                    PyPlot.grid()
                end
                
                
                if num_subplots > 2
                    subplot(num_subplots*100 + 13)
                    #plot((10^9)*X[:],U_bulk[1,:],label="phi (V)")
                    plot(time_range,phi_range_full,label="phi_S (V)")
                    plot(time_range,U0_range,label="y(0)")
                    plot(time_range,Ub_range,label="y_s")
                    PyPlot.xlabel("t (s)")
                    PyPlot.legend(loc="best")
                    PyPlot.grid()
                end
                
                
                
                if false
                #if istep_cv_start > -1
                    
                    nodes = check_nodes_whole()
                    U_orig = zeros(0)
                    I_orig = zeros(0)
                    for i in nodes
                        append!(U_orig,i[1])
                        append!(I_orig,i[3])
                    end
                
                    subplot(num_subplots*100 + 14)
                    plot(phi_range[istep_cv_start:end].-phi0, (Is_range + Ib_range + r_range + Ibb_range)[istep_cv_start:end] ,label="sim")
                    plot(U_orig, I_orig ,label="exp")
                    PyPlot.xlabel("E (V)")
                    PyPlot.ylabel("I (A)")
                    PyPlot.xlabel("t (s)")
                    PyPlot.legend(loc="best")
                    PyPlot.grid()
                end
                
                pause(1.0e-10)
            end
            
            
            istep+=1
            
            if state=="relaxation"
                relax_counter += 1
                #println("relaxation ... ",relax_counter/sample*100,"%")
            else
                phi+=voltrate*dir*tstep
            end
        end
    
        
        
        
        
	if pyplot
		cv_range = (istep_cv_start+1):length(phi_range)
		
		PyPlot.clf()
		PyPlot.figure(figsize=(5,5))
		if dlcap
		  #plot(phi_range[cv_range].-phi0,( Ib_range[cv_range] )/voltrate,"blue", label="bulk")
		  ##plot(phi_range[cv_range].-phi0,( Ibb_range[cv_range])/voltrate,label="bulk_grad")
		  #plot(phi_range[cv_range].-phi0,( Is_range[cv_range] )/voltrate,"green", label="surf")
		  #plot(phi_range[cv_range].-phi0,( r_range[cv_range]  )/voltrate,"red", label="reac")
		else
		  plot(phi_range[cv_range].-phi0, Ib_range[cv_range] ,"blue", label="bulk")
		  #plot(phi_range[cv_range].-phi0, Ibb_range[cv_range] ,label="bulk_grad")
		  plot(phi_range[cv_range].-phi0, Is_range[cv_range] ,"green",label="surf")
		  plot(phi_range[cv_range].-phi0, r_range[cv_range] ,"red",label="reac")
		end
		if dlcap
                        PyPlot.xlabel("nu (V)")
                        PyPlot.ylabel(L"Capacitance (F/m$^2$)")  
                        PyPlot.legend(loc="best")
                        PyPlot.xlim(-0.5, 0.5)
                        PyPlot.ylim(0, 5)
                        PyPlot.grid()
                        PyPlot.show()
                        PyPlot.pause(10)
                        
                        PyPlot.clf()
                        plot(phi_range[cv_range].-phi0,( (Ib_range+Is_range+r_range)[cv_range]  )/voltrate,"brown", label="total")
                        PyPlot.xlabel("nu (V)")
                        PyPlot.ylabel(L"Capacitance (F/m$^2$)") 
                        PyPlot.legend(loc="best")
                        PyPlot.xlim(-0.5, 0.5)
                        PyPlot.ylim(0, 5)
                        PyPlot.grid()
                        PyPlot.show()
                        #PyPlot.pause(10)
                    end
		
		

		
		subplot(222)
		if dlcap
		    #cbl, cs = direct_capacitance(parameters, collect(float(low_bound):0.001:float(upp_bound)))
		    #plot(collect(float(low_bound):0.001:float(upp_bound)), (cbl+cs), label="tot CG") 
		    #plot(collect(float(low_bound):0.001:float(upp_bound)), (cbl), label="b CG") 
		    #plot(collect(float(low_bound):0.001:float(upp_bound)), (cs), label="s CG") 
		    #plot(phi_range[cv_range].-phi0, ((Is_range + Ib_range + r_range + Ibb_range)[cv_range])/voltrate ,label="rescaled total current")# rescaled by voltrate
		else
		    plot(phi_range[cv_range].-phi0, ((Is_range + Ib_range + r_range + Ibb_range)[cv_range]) ,label="total current")
		end
		PyPlot.xlabel("E (V)")
		PyPlot.legend(loc="best")
		PyPlot.grid()
		
		subplot(223)
		#plot(phi_range, r_range ,label="spec1")
		plot(time_range,phi_range_full,label="phi_s (V)")        
		plot(time_range,U0_range,label="y(0)")
		plot(time_range,Ub_range,label="y_s")
		PyPlot.xlabel("t (s)")
		PyPlot.legend(loc="best")
		PyPlot.grid()
		
		subplot(224, facecolor="w")
		height=0.0
		shift=0.0
		swtch=false
		for name in fieldnames(typeof(parameters))
		    if (string(name) == "A0" || swtch)
		        swtch = true
		        value = @sprintf("%.6g", parse(Float64,string(getfield(parameters,name))))
		        linestring = @sprintf("%s: %s", name, value)
		        PyPlot.text(0.01+shift, 0.95+height, linestring, fontproperties="monospace")
		        height+=-0.05
		        if string(name) == "e0" 
		            shift+=0.5
		            height=0.0
		        end
		        if string(name) == "A"
		            PyPlot.text(0.01+shift, 0.95+height, " ", fontproperties="monospace")
		            height+=-0.05
		        end
		    end
                    end
		parn = ["verbose" ,"pyplot", "width", "voltametry", "voltrate", "low_bound", "upp_bound", "sample", "phi0"]
		parv =[verbose ,pyplot, width, voltametry, voltrate, low_bound, upp_bound, sample, @sprintf("%.6g",phi0)]
		for ii in 1:length(parn)
			linestring=string(parn[ii],": ",parv[ii])
		        PyPlot.text(0.01+shift, 0.95+height, linestring, fontproperties="monospace")
		        height+=-0.05
            	end
            	#plot(phi_range, r_range ,label="spec1")
            	#PyPlot.legend(loc="best")
            	#PyPlot.grid()
        end
        #plot(phi_range, r_range ,label="spec1")
        #PyPlot.legend(loc="best")
        #PyPlot.grid()

        
        

        if par_study_bool
            cv_range = (istep_cv_start+1):length(phi_range)
            cbl, cs = direct_capacitance(parameters, collect(float(low_bound):0.001:float(upp_bound)))
            plot(collect(float(low_bound):0.001:float(upp_bound)), (cs), label="b CG")
            #plot(phi_range[cv_range].-phi0, ((Is_range + Ib_range + r_range + Ibb_range)[cv_range])/voltrate ,label="rescaled total current")# rescaled by voltrate
            plot(phi_range[cv_range].-phi0, (( Is_range )[cv_range])/voltrate ,label="rescaled total current")# rescaled by voltrate
            PyPlot.legend(loc="best")
        end

        
        if save_files
            out_name=string(
            "A0",@sprintf("%.0f",prms_in[1]),
            "_R0",@sprintf("%.0f",prms_in[2]),
            "_GA",@sprintf("%.0f",prms_in[3]),
            "_GR",@sprintf("%.0f",prms_in[4]),
            "_be",@sprintf("%.0f",prms_in[5]),
            "_A",@sprintf("%.0f",prms_in[6]),
            "_vrate",@sprintf("%.2g",voltrate),
            )


            CSV.write(string("./data/",out_name,".csv"),out_df)
            if pyplot
                PyPlot.savefig(string("./images/",out_name,".png"))
            end
        end
        #######################################################
        #######################################################
    end
        
    if fitting
        return CV_get_error(
            phi_range[istep_cv_start:end].-phi0,
            ((Is_range + Ib_range + r_range + Ibb_range)[istep_cv_start:end]),
            #check_nodes_short()
            check_nodes_whole(),
            opt_pyplot=true
            )
    end
end

function iter_dtstep()
    for i in 1.0e-6*[2]
        run_new(voltametry=true, dlcap=true, save_files=true, pyplot=true, dtstep_in=i)
    end
end

function iter_voltrate()
    for i in [10.0^i for i in collect(0:1:3)]
        run_new(voltametry=true, dlcap=true, save_files=true, pyplot=true, voltrate=i, sample=2500, bound=.9, A0_in=-2)
    end
end

#function par_study()
#  err_counter::Int32 = 0
#  good_counter::Int32 = 0
#  all_counter::Int32 = 0
#  for A0_i in [5] #[-10, -5, 0, 5, 10, 15, 20]
#    for DGA_i in [0] #[-100000, -1000, -100, 0, 100, 1000, 100000]
#      for DGR_i in [4] #[-6, -4, -2, 0, 2, 4, 6, 8, 10]
#        for R0_i in [0] #[-20, -10, -0, 10, 20]
#          all_counter = all_counter + 1
#          println(string(" <><><><><><><><><><><><> all_counter <><><> ",all_counter," of ", 7*6*5*5))
#          #try
#            #run_open(n=100, sample=200, verbose=false, pyplot=false, width=2.0e-9, voltametry=true, voltrate=0.005, bound=.7, A0_in=A0_i, #DGA_in=DGA_i, DGR_in=DGR_i, R0_in=R0_i)
#          run_new(sample=10, hexp=-9., voltametry=true, voltrate=.005, bound=0.55, phi0=-0.25, pyplot=false, A0_in=A0_i, DGA_in=DGA_i, DGR_in=DGR_i, #R0_in=R0_i)
#          good_counter = good_counter + 1
#          #catch
#            #err_counter = err_counter + 1
#          #end
#        end
#      end
#    end
#  end
#  println(string("err_counter / good_counter >>>>>>>>>>>>>>>>> ",err_counter,"  /  ",good_counter))
#end




function nu_iter()
    for nu_in in [0.1, 0.5, 0.9]
        run_new(print_bool=false, fitting=false, voltametry=true, dlcap=true, pyplot=false, voltrate=0.00001, sample=50, upp_bound=1, low_bound=-1, 
            prms_in=[-2, 0, 0.2, -0.2, 0.6074566741435283, 0.1], width=1e-8, nu_in=nu_in, par_study_bool=true)
    end
end


function prepare_prms(mask, x0, x)
    prms = zeros(0)
    xi = 1
    for i in collect(1 : 1 :length(mask))
        if convert(Bool,mask[i])
            append!(prms, x[xi])
            xi += 1
        else
            append!(prms, x0[i])
        end
    end
    return prms
end

function my_optimize()
    function rosenbrock(x)
	#[1 - x[1], 100 * (x[2]-x[1]^2)]
# 	#[1 - x[1]]
	[1]
    end

    function to_optimize(x) 
        #err = run_new(print_bool=false, fitting=true, voltametry=true, pyplot=false, voltrate=0.005, sample=8, bound=0.41, 
        #    prms_in=x)
        prms = prepare_prms(mask, x0, x)
        print(" >> mask = ",mask)
        print(" || prms = ",prms)
        
        #err = run_new(print_bool=false, fitting=true, voltametry=true, pyplot=true, voltrate=0.005, sample=50, upp_bound=0.474, low_bound=-0.429, 
        #    prms_in=prms)
            
        err = run_new(print_bool=true, fitting=true, voltametry=true, dlcap=false, pyplot=true, voltrate=0.001, sample=10, upp_bound=0.55, low_bound=-0.548, 
            prms_in=prms, width=0.45e-3)
        
        println(" || err =", err)
        return [err]
    end
    #x0 = zeros(2)
    #optimize(rosenbrock, x0, LevenbergMarquardt())
    
    lower_bounds = [-20, -20, -1.2, -1.2, 0.1, 0.1]
    upper_bounds = [15, 22, 0.2, 0.2, 0.9, 1.0]
    
    #x0 = [3.1149, -9.59917, 0.498534, -0.597552, 0.724949, 1.17139]
    #x0 = [3.40882, -4.1195, 0.0, 0.0, 0.5, 1.17139]
    #x0 = [4.42991, 19.03254, 0.0, 0.0, 0.5, 0.301]
    #x0 = [4.42991, 20.03254, 0.0, 0.0, 0.5, 0.151]
    
    #x0 = [9.00137, 20.1747, 0.5, 0.0, 0.5, 0.206002]
    x0 = [2.3, 19.5, 1, -1, 0.6, 1.2]
    #x0 = [2.32407, 18.4458, 1.0, -1.0, 0.769217, 1.03349]
    #x0 = [2.34223, 18.4039, 1.0, -1.0, 0.72743, 0.95021] # quite good fit <<<<<<<<<<
    #x0 = [2.34223, 20.8039, 1.0, -1.0, 0.52743, 0.25021] # hand fit <<< usable
    #x0 = [2.74223, 19.7039, 1.0, -1.0, 0.58, 0.25021]
    #x0 = [2.44223, 20.5039, 1.0, -1.0, 0.61, 0.25021] # not bad :)) <<<<<<<<<
    
    # err metric <- check_nodes_long()
    x0 = [2.54223, 20.5039, 1.0000000, -1.000000, 0.500000, 0.250210]  # by hand
    x0 = [2.54223, 20.5039, 1.0000000, -1.000000, 0.632507, 0.250210] # fitted beta << GOOD << err =0.006814132871406775
    x0 = [2.54223, 20.5039, 1.0000000, -1.000000, 0.632507, 0.253287] # fitted A  << err =0.006807257238292433
    x0 = [2.54223, 20.5033, 1.0000000, -1.000000, 0.632507, 0.253287] # fitted R0 << err =0.0068072339219288356
    x0 = [2.55263, 20.5033, 1.0000000, -1.000000, 0.632507, 0.253287] # fitted A0 << err =0.006745444782481952
    x0 = [2.58082, 20.4100, 1.0000000, -1.000000, 0.632507, 0.253287] # fitted A0, R0 << err =0.006574359568544145
    x0 = [2.58082, 20.4100, 0.0905748, -0.708014, 0.632507, 0.253287] # fitted DGA, DGR << err =0.006573859072513787
    x0 = [2.70077, 20.5264, 0.0905748, -0.708014, 0.598817, 0.143269] # fittet 110011 << err =0.005685020802399199
    
    # err metric <- check_nodes_whole()
    x0 = [2.74851, 20.5631, 0.0905748, -0.708014, 0.605159, 0.105409] # fitting... 110011 << err =0.0054800474768966585
    x0 = [2.73650, 20.6063, 0.0905748, -0.708014, 0.607443, 0.100000] # fitting... 110011 << err =0.005415825589421705
    x0 = [2.73645, 20.6064, 0.0905748, -0.708014, 0.607457, 0.100000] # fitted 110011 <<  err =0.0054158249335496105
    x0 = [2.736451985137371, 20.606423236896422, 0.0905748, -0.708014, 0.6074566741435283, 0.1] # ACURATE FINALL FIT
    # >> x0corr = [21.71975544711280, 20.606423236896422, 0.0905748, -0.708014, 0.6074566741435283, 0.1] # ACURATE FINALL FIT
    #x0 = [6.736451985137371, 24.606423236896422, 0.0905748, -0.708014, 0.6074566741435283, 0.1]
    #x0 = [2.736451985137371, 20.606423236896422, 0.0905748, -0.1508014, 0.6074566741435283, 0.1]
    #x0 = [1.736451985137371, 15.106423236896422, 0.0905748, -0.108014, 0.6074566741435283, 0.1] # ploted to slides
    # >> x0corr = [20.71975544711280, 15.106423236896422, 0.0905748, -0.108014, 0.6074566741435283, 0.1] # ploted to slides
    
    mask = [0, 0, 1, 1, 0, 0] # determining, which parametr should be fitted
    
    x0M = zeros(0)
    lowM = zeros(0)
    uppM = zeros(0)
    for i in collect(1 : 1 : length(mask))
        if convert(Bool,mask[i])
            append!(x0M, x0[i])
            append!(lowM, lower_bounds[i])
            append!(uppM, upper_bounds[i])
        end
    end
    
    PyPlot.close()
    to_optimize(x0M)
    #println(optimize(to_optimize, x0M, lower=lowM, upper=uppM, Δ=1000, f_tol=1.0e-14, g_tol=1.0e-14, LevenbergMarquardt()))
    #optimize(to_optimize, x0M, Dogleg())
    return
end

function cap_nodes(T)
    if T == 848
        return [
            [-0.24979,  1,  3.65607],
            [-0.22393,  1,  3.92012],
            [-0.2001,   1,  4.3482],
            [-0.17515,  1,  5.11678],
            [-0.15058,  1,  5.82232],
            [-0.12518,  1,  6.33863],
            [-0.10049,  1,  6.50182],
            [-0.07534,  1,  6.41274],
            [-0.0505,   1,  5.90745],
            [-0.02553,  1,  4.8472],
            [-0.00073,  1,  4.51849],
            [0.02527,   1,  4.17713],
            [0.05048,   1,  3.79796]
        ]
    end
    if T == 798
        return [
            [-0.17426,  1,  2.96],
            [-0.12419,  1,  3.76563],
            [-0.10035,  1,  4.15587],
            [-0.07537,  1,  4.7731],
            [-0.04981,  1,  4.63355],
            [-0.02539,  1,  4.20395],
            [0.0006,    1,  3.91304],
            [0.0246,    1,  3.60958],
            [0.05102,   1,  3.17992]
        ]
    end

    if T == 748
        return [
            [-0.09908,  1,  2.08736],
            [-0.07522,  1,  2.3767],
            [-0.04894,  1,  2.56506],
            [-0.02427,  1,  2.79131],
            [0.00049,   1,  2.6644],
            [0.02642,   1,  2.61313],
            [0.0516,    1,  2.3727]
        ]
    end
    
    println("Err: cap_nodes: wrong T")
    return "wrong T"
end


function cap_optimize()

    function to_optimize(x)
        #println("x = ",x)
        prms = prepare_prms(mask, x0, x)
        print(" >> mask = ",mask)
        print(" || prms = ",prms)
        
        parameters = YSZParameters()
        
        parameters.nu = prms[1]
        parameters.nus = prms[2]
        parameters.DGA = prms[3]*parameters.e0
        parameters.m_par = prms[4]
        parameters.ms_par = prms[5]
        
        parameters = YSZParameters_update(parameters)
        
        phi_range_cap = collect(-1:0.01:1)
        cbl, cs = direct_capacitance(parameters,phi_range_cap)
        
        err = CV_get_error(phi_range_cap,cbl+cs, cap_nodes(parameters.T))
        
        #plot(phi_range_cap, cs, label="tot_capacitance")
        #PyPlot.legend(loc="best")
        #PyPlot.grid()
        
        
        
        println(" || err =", err)
        return [err]
    end
    

    
    lower_bounds = [0, 0, -1.2, 0.1, 0.1]
    upper_bounds = [1, 1, 1.0, 3, 3]
    
    mask = [1, 1, 1, 1, 1]
    # x0 = [nu, nus, DGA, m, ms]
    x0 = [0.1, 0.9, -0.5, 2, 2]
    x0 = [0.1, 0.7, -0.2, 2, 0.5]
    
    x0M = zeros(0)
    lowM = zeros(0)
    uppM = zeros(0)
    
    for i in collect(1 : 1 : length(mask))
        if convert(Bool,mask[i])
            append!(x0M, x0[i])
            append!(lowM, lower_bounds[i])
            append!(uppM, upper_bounds[i])
        end
    end
    
    opt_pyplot=false
    to_optimize(x0M)
        
    println(optimize(to_optimize, x0M, lower=lowM, upper=uppM, Δ=1000, x_tol = 1.0e-14, f_tol=1.0e-14, g_tol=1.0e-14, LevenbergMarquardt()))
    return
end

function nu_anal_iter()
    PyPlot.close()
    iter_pyplot = true
    iter_save_file_bool = true

    
    
    phi_range_cap = collect(-0.5: 0.01: 0.5)

    
    laast = [0.47, 0.44, -0.2, 0.15, 60.0]
    

    
    # .........[nu  , nus, DGA*eV, ms,   chi]
    # 748K ->> [0.85, 0.21, -0.14, 0.05, 27.0]
    # 798K ->> [0.55, 0.22, -0.17, 0.12, 27.0]
    # 848K ->> [0.07, 0.51, -0.18, 0.3, 27.0]

    parameters = YSZParameters()
    printfields(parameters)
    
    DATA = []
    T748list = []
    T798list = []
    T848list = []
    sum_list = []
    sum = 0

    overall_min = 666
    overall_ms = -1
    
    ms_range = collect(0.05: 0.002: 0.65)
    
    for l in ms_range  # for >>> ms <<<
        
        
        for T_in in collect(748 : 50 : 848)
        
            err_min = 666
            prms_min = [0, 0, 0, 0, 0]
            cscdl_min = phi_range_cap
            
            #parameters.T = T_in
            parameters.T = 848
            
            
            nodes = cap_nodes(parameters.T)
            
            for i in collect(0.0: 0.01: 0.93)
            #for i in collect(0.0: 0.01: 0.93)
                for j in collect(0.1 : 0.01: 0.93)
                    #println(" ... j ",j)   
                    for k in collect(-0.5: 0.02: 0.5)
                        #for cc in collect(20: 3 :120)
                        for cc in collect(27: 2 :27)
                            
                            prms=[i,j,k,l,cc]
                            if iter_pyplot
                                prms=[0.85, 0.1, 0.05, 0.15, 27.0]
                                prms=[0.89, 0.15, 0.15, 0.05, 121.0]
                                prms=[0.86, 0.7, 0.1, 0.05, 114.0]
                                prms=[0.63, 0.21, 0.05, 0.25, 27.0]
                                prms=[0.07, 0.51, -0.18, 0.3, 27.0]

                                if parameters.T == 748
                                    #prms = [0.85, 0.21, -0.14, 0.05, 27.0]
                                    prms = [0.85, 0.85, -0.14, 0.262, 27.0]
                                end
                                if parameters.T == 798
                                    #prms = [0.55, 0.22, -0.17, 0.12, 27.0]
                                    prms = [0.57, 0.64, -0.16, 0.262, 27.0]
                                end
                                if parameters.T == 848
                                    #prms = [0.07, 0.51, -0.18, 0.3, 27.0]
                                    prms = [0.07, 0.44, -0.18, 0.262, 27.0]
                                end
                            end
                            
                            parameters.nu = prms[1]
                            parameters.nus = prms[2]
                            parameters.DGA = parameters.e0*prms[3]
                            parameters.m_par = 2
                            parameters.ms_par = prms[4]
                            parameters.chi = prms[5]
                            
                            
                            
                            parameters = YSZParameters_update(parameters)
                            
                            
                            cbl, cs = direct_capacitance(parameters,phi_range_cap)

                            
                            
                            if iter_pyplot
                                U_orig = zeros(0)
                                I_orig = zeros(0)
                                for i in nodes
                                    append!(U_orig,i[1])
                                    append!(I_orig,i[3])
                                end
                                
                                plot(phi_range_cap, cs + cbl, label=string("prms ",prms, " ... T = ",parameters.T,"K"))
                                plot(U_orig,I_orig, label="exp")
                                PyPlot.legend(loc="best")
                                PyPlot.grid()
                                
                                PyPlot.draw()
                                PyPlot.show()
                                
                            end
                            
                            
                            err = CV_get_error(phi_range_cap,cbl+cs, nodes, opt_pyplot=false)
                            #println(i," ",j," ",err)
                            if err < err_min
                                err_min = err
                                prms_min = prms
                                cscdl_min =  cs + cbl
                            end
                            
                            #push!(DATA,[prms, err])
                            
                            if iter_save_file_bool
                                iter_out_df = DataFrame(phi = Float64[], C = Float64[])
                                for i in collect(1 : 1 : length(phi_range_cap))
                                    push!(iter_out_df,[phi_range_cap[i]   (cs+cbl)[i]])
                                end
                                out_name=string(
                                    "Cap_fit_T",@sprintf("%.0f",parameters.T),
                                    "_nu",@sprintf("%.2f",parameters.nu),
                                    "_nus",@sprintf("%.2f",parameters.nus),
                                    "_DGA_in",@sprintf("%.2f",prms[3]),
                                    "_ms",@sprintf("%.2f",prms[4])
                                    )
                                CSV.write(string("./data/",out_name,".csv"),iter_out_df)
                            end
                            
                            if iter_pyplot
                                PyPlot.pause(2)
                                return
                            end 
                        end
                    end
                    
                end
            
            
                #println("i ",i, " >> err_min ",err_min, "      prms_min ",prms_min)
    
    
            
                if false
                    U_orig = zeros(0)
                    I_orig = zeros(0)
                    for ii in nodes
                        append!(U_orig,ii[1])
                        append!(I_orig,ii[3])
                    end

                    plot(phi_range_cap, cscdl_min, label=string("prms_min ",prms_min, " ... T = ",parameters.T,"K"))
                    plot(U_orig,I_orig, label="exp")
                    PyPlot.legend(loc="best")
                    PyPlot.grid()

                    PyPlot.draw()
                    PyPlot.show()
                    PyPlot.pause(0.1)    
                end
            end
            
            println("ms ",l,"  T ",parameters.T," err_min ",err_min,"  prms_min ",prms_min)
            if parameters.T == 748
                push!(T748list,err_min)   
            end
            if parameters.T == 798
                push!(T798list,err_min)
            end
            if parameters.T == 848
                push!(T848list,err_min)
            end
        end
        sum = T748list[end]+T798list[end]+T848list[end]
        
        push!(sum_list,sum)
        if sum < overall_min
            overall_min = sum
            overall_ms = l
        end
        
        println(" ------ ")
    end
    #println(length(DATA))
    #println(DATA[1])
    #println(DATA[2][1])
    #println("err_min ",err_min, "      prms_min ",prms_min)
    #PyPlot.pause(100)
    
    subplot(411)
    plot(ms_range, T748list, label="748K")
    PyPlot.xlabel("ms (1)")
    PyPlot.ylabel("Error")
    PyPlot.legend(loc="best")
    PyPlot.grid()
    
    subplot(412)
    plot(ms_range, T798list, label="798K")
    PyPlot.xlabel("ms (1)")
    PyPlot.ylabel("Error")
    PyPlot.legend(loc="best")
    PyPlot.grid()
    
    subplot(413)
    plot(ms_range, T848list, label="848K")
    PyPlot.xlabel("ms (1)")
    PyPlot.ylabel("Error")
    PyPlot.legend(loc="best")
    PyPlot.grid()
    
    subplot(414)
    plot(ms_range, sum_list, label="sum")
    PyPlot.xlabel("ms (1)")
    PyPlot.ylabel("Error")
    PyPlot.legend(loc="best")
    PyPlot.grid()
    
    println("min ms >>> ",overall_ms,"   ...   min sum >>> ",overall_min)
    
    PyPlot.draw()
    PyPlot.show()
    PyPlot.pause(1000) 
    println("<< end >>")
end

# TODO //////////
#   [x] fix the problem with non-zero current during relaxation
#   [x] eliminate parameter "n" and "h" used during calculation of dphiB
#   [ ] validate the code to analytic solution with apropriate right-hand-side
