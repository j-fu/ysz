"""
YSZ example cloned from iliq.jl of the TwoPointFluxFVM package.
"""

using Printf
using TwoPointFluxFVM
using PyPlot
using DataFrames
using CSV


mutable struct YSZParameters <:TwoPointFluxFVM.Physics
    TwoPointFluxFVM.@AddPhysicsBaseClassFields

    # to fit
    A0::Float64   # surface adsorption coefficient 
    R0::Float64 # exhange current density [A/m^2]
    DGA::Float64 # difference of gibbs free energy of adsorbtion
    DGR::Float64 # difference of gibbs free energy of electrochemical reaction
    beta::Float64 # symmetry of the reaction
    A::Float64 # activation energy of the reaction
    
    # fixed
    DD::Float64   # diffusion coefficient [m^2/s]
    pO::Float64 # O2 partial pressure [bar]
    T::Float64      # Temperature [K]
    nu::Float64    # ratio of immobile ions, \nu [1]
    nus::Float64    # ratio of immobile ions on surface, \nu [1]
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
    
    
    this.A0=1e-1#.e3
    this.R0=1.0e-2
    this.DGA=-1.0e5
    this.DGR=0.0e-2
    this.beta=0.5
    this.A=2.
    
    
    this.DD=1.5658146540360312e-11  # fitted to conductivity 0.063 S/cm ... TODO reference
    this.pO=1.                      # O2 atmosphere 
    this.T=1073                     
    this.nu=0.3                     # assumption
    this.nus=0.3                    # assumption
    this.x_frac=0.08                # 8% YSZ
    this.chi=6.e0                   # from relative permitivity e_r = 6 ... TODO reference
    this.m_par = 2                  
    this.ms_par = this.m_par        
    
    
    this.vL=3.35e-29
    this.areaL=(this.vL)^0.6666
    this.e0   = 1.602176565e-19  #  [C]
    this.eps0 = 8.85418781762e-12 #  [As/(Vm)]
    this.kB   = 1.3806488e-23  #  [J/K]
    this.N_A  = 6.02214129e23  #  [#/mol]
    this.zA  = -2;
    this.mO  = 16/1000/this.N_A  #[kg/#]
    this.mZr = 91.22/1000/this.N_A #  [kg/#]
    this.mY  = 88.91/1000/this.N_A #  [kg/#]
    this.zL  = 4*(1-this.x_frac)/(1+this.x_frac) + 3*2*this.x_frac/(1+this.x_frac) - 2*this.m_par*this.nu
    this.y0  = -this.zL/(this.zA*this.m_par*(1-this.nu))
    this.ML  = (1-this.x_frac)/(1+this.x_frac)*this.mZr + 2*this.x_frac/(1+this.x_frac)*this.mY + this.m_par*this.nu*this.mO
    # this.zL=1.8182
    # this.y0=0.9
    # this.ML=1.77e-25    
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
                this.DGA/(this.kB*this.T) 
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
                this.DGA/(this.kB*this.T) 
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
    PHI = -domain#collect(-bound:0.001:bound) # PHI = phi_B-phi_S, so domain as phi_S goes with minus
    #
    yB = -this.zL/this.zA/this.m_par/(1-this.nu);
    X  = yB/(1-yB)*exp.(this.zA*this.e0/this.kB/this.T*PHI)
    y  = X./(1.0.+X)
    #
    nF = this.e0/this.vL*(this.zL .+ this.zA*this.m_par*(1-this.nu)*y)
    F  = sign.(PHI).*sqrt.(
          2*this.e0/this.vL/this.eps0/(1.0+this.chi).*(
            this.zL.*PHI .+ this.kB*this.T/this.e0*this.m_par*(1-this.nu)*log.(
              (1-yB).*(X .+ 1.0)
             )
           )
         );
    #
    Y  = yB/(1-yB)*exp.(this.DGA/this.kB/this.T .+ this.zA*this.e0/this.kB/this.T*PHI);
    #
    CS = this.zA^2*this.e0^2/this.kB/this.T*this.ms_par/this.areaL*(1-this.nus)*Y./(1.0.+Y).^2;
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
    B = exp( - (this.DGA + this.DGR) / (this.kB * this.T))*this.pO^(1/2.0)
    println((this.DGA + this.DGR))
    println((this.kB * this.T))
    println(- (this.DGA + this.DGR) / (this.kB * this.T))
    println(this.pO^(1/2.0))
    println(B)
    println(B/(1+B))
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

function debug(this::YSZParameters, u, bu)
    println("Debug ////////////////////////////////////////// ")
    println("y~ys ",log(u[iy]*(1-bu[1])) - log(bu[1]*(1-u[iy])))
    
    electroR=electroreaction(this,bu)
    f= this.A0*(this.mO/this.areaL)*
            (
                this.DGA/(this.kB*this.T) 
                +    
                log(u[iy]*(1-bu[1]))
                - 
                log(bu[1]*(1-u[iy]))
                
            )
    println("f = ",f, "       elreac = ",electroR)
end
###########################################################
###########################################################

function run_new(;n=15, hexp=-8, verbose=false ,pyplot=false, width=10.0e-9, voltametry=false, dlcap=false, save_files=false, voltrate=0.005, phi0=0.0, bound=1.0, sample=50, A0_in=-2, R0_in=2, DGA_in=1.5, DGR_in=-1.0, beta_in=0.5, A_in = -1 )

    # A0_in \in [-6, 6]
    # R0_in \in [-6, 6]
    # DGA_in \in [-1, 0]
    # DGR_in \in [-2, 2]
    # beta_in \in [0,1]
    # A_in \in [-1, 1]
    #
    # AREA OF THE ELECTROLYTE CUT
    AreaEllyt = 0.000201 * 0.6      # m^2     (1 - porosity)
    width_Ellyt = 0.00045           # m     width of the half-cell
    if dlcap
        AreaEllyt = 1      # m^2     (1 - porosity)
        println("dlcap > area = 1")
    end
    # 
    h=width/convert(Float64,n)
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
    eV = 1.602e-19   # electronvolt [J]
    
    parameters=YSZParameters()
    # for parametric study
    parameters.A0 = 10.0^A0_in      # [1 / s]
    parameters.R0 = 10.0^R0_in      # [1 / m^2 s]
    if dlcap
        parameters.R0 = 0
        println("dlcap > R0= ",parameters.R0)
    end
    parameters.DGA = DGA_in * eV    # [J]
    parameters.DGR = DGR_in * eV    # [J]
    parameters.beta = beta_in       # [1]
    parameters.A = 10.0^A_in        # [1]
    
    #
    parameters.storage=storage!
    parameters.flux=flux!
    parameters.reaction=reaction!
    parameters.breaction=breaction!
    parameters.bstorage=bstorage!
    #
    printfields(parameters)
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
    if dlcap
        phi0 = 1e-5
        println("dlcap > phi0= ", phi0)
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
    control.damp_initial=0.01
    control.damp_growth=2
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
            println("t = ",t)
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
        istep=0
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
        
        MY_Ir_range=zeros(0)
        MY_Itot_range=zeros(0)
        out_df = DataFrame(t = Float64[], U = Float64[], I = Float64[])
        
        cv_cycles = 2
        relaxation_length = 2    # how many "samples" should relaxation last
        relax_counter = 0
        istep_cv_start = 0
        time_range = zeros(0)  # [s]

        print("calculating linear potential sweep\n")
        direction_switch = 0
        dtstep=1e-6
        tstep=bound/voltrate/sample      
        if dtstep > tstep
            dtstep=0.05*tstep
            print("dtstep refinement: ")
        end
        if phi0 > 0
            dir=1
        else
            dir=-1
        end
        
        if pyplot
            PyPlot.close()
            PyPlot.ion()
            PyPlot.figure(figsize=(10,8), )
        end
        
        println("phi_equilibrium = ",phi0)
        state = "ramp"
        println("ramp ......")
        while state != "cv_is_off"
            if state=="ramp" && ((dir==1 && phi > phi0) || (dir==-1 && phi < phi0))
                phi = phi0
                state = "relaxation"
                println("relaxation ... ")
            end            
            if state=="relaxation" && relax_counter == sample*relaxation_length
                relax_counter += 1
                state="cv_is_on"
                istep_cv_start = istep
                dir=1
                print("cv ~~~ direction switch: ")
            end                            
            if state=="cv_is_on" && (phi <= -bound+phi0 || phi >= bound+phi0)
                dir*=(-1)
                # uncomment next line if phi should NOT go slightly beyond limits
                #phi+=2*voltrate*dir*tstep
            
                direction_switch +=1
                print(direction_switch,", ")                
            end            
            if state=="cv_is_on" && (dir > 0) && (phi > phi0 + 0.000001) && (direction_switch >=2*cv_cycles)
                state = "cv_is_off"
            end
            
            
            # tstep to potential phi
            sys.boundary_values[iphi,1]=phi
            U=solve(sys,inival,control=control,tstep=tstep)
            inival.=U
            Qb=integrate(sys,reaction!,U) # - \int n^F
            dphiB=parameters.eps0*(1+parameters.chi)*(0 - bulk_unknowns(sys,U)[end][1])/h 
            y_bound=boundary_unknowns(sys,U,1)
            Qs= -(parameters.e0/parameters.areaL)*parameters.zA*y_bound*parameters.ms_par*(1-parameters.nus) # - n^F_s

                 
            # dtstep to potential (phi + voltrate*dtstep)
            sys.boundary_values[iphi,1]=phi+voltrate*dir*dtstep
            Ud=solve(sys,U,control=control,tstep=dtstep)
            Qbd=integrate(sys,reaction!,Ud)
            dphiBd = parameters.eps0*(1+parameters.chi)*(0 - bulk_unknowns(sys,Ud)[end][1])/h
            yd_bound=boundary_unknowns(sys,Ud,1)
            Qsd= -(parameters.e0/parameters.areaL)*parameters.zA*yd_bound*parameters.ms_par*(1-parameters.nus)

            # time derivatives
            dphiBdt = (-dphiB + dphiBd)/dtstep
            Ibb = -dphiBdt
            Ib=(Qbd[iphi] - Qb[iphi])/dtstep #- dphiBdt
            Is=(Qsd[1] - Qs[1])/dtstep

            # reaction average
            reac = -2*electroreaction(parameters, y_bound)
            reacd = -2*electroreaction(parameters, yd_bound)
            Ir=0.5*(reac + reacd)

            #############################################################
            #multiplication by area of electrode I = A * ( ... )
            Ibb = Ibb*AreaEllyt
            Ib = Ib*AreaEllyt
            Is = Is*AreaEllyt
            Ir = Ir*AreaEllyt
            #
            
            if verbose
                #@printf("t = %g     U = %g   state = %s  reac = %g  \n", istep*tstep, phi, state, Ir)
                #debug(parameters,U, U_bound)
            
            end
            U_bulk=bulk_unknowns(sys,U)
            U_bound=boundary_unknowns(sys,U,1)
            
            # storing data
            append!(time_range,tstep*istep)
            
            append!(U0_range,U_bulk[iy,1])
            append!(Ub_range,U_bound[1,1])
            append!(phi_range_full,phi)
            
            append!(MY_Ir_range,Ir)
            append!(MY_Itot_range,Ibb+Is+Ib+Ir) 

            append!(phi_range,phi)
            append!(Ib_range,Ib)
            append!(Is_range,Is)
            append!(Ibb_range,Ibb)
            append!(r_range, Ir)
               
            if state=="cv_is_on"
                push!(out_df,[(istep-istep_cv_start)*tstep   phi-phi0    Ibb+Is+Ib+Ir])
            end
            
            
            
            ##### my plotting                  
            if pyplot && istep%10 == 0
                PyPlot.clf()
                subplot(311)
                plot((10^9)*X[:],U_bulk[iphi,:],label="phi (V)")
                plot((10^9)*X[:],U_bulk[iy,:],label="y")
                plot(0,U_bound[1,1],"go", markersize=3, label="y_s")
                l_plot = 1.0
                PyPlot.xlim(-0.01*l_plot, l_plot)
                PyPlot.ylim(-0.5,1.1)
                PyPlot.xlabel("x (nm)")
                PyPlot.legend(loc="best")
                PyPlot.grid()
                
                subplot(312)
                
                #plot((10^9)*X[:],U_bulk[1,:],label="phi (V)")
                
                plot(time_range,phi_range_full,label="phi_S (V)")
                plot(time_range,U0_range,label="y(0)")
                plot(time_range,Ub_range,label="y_s")
                PyPlot.legend(loc="best")
                PyPlot.grid()
                
                subplot(313)
                plot(time_range,Ib_range, label = "I_bulk")
                plot(time_range,Ibb_range, label = "I_bulkgrad")
                plot(time_range,Is_range, label = "I_surf")
                plot(time_range,r_range, label = "I_reac")
                PyPlot.ylabel("I (A)")
                PyPlot.xlabel("t (s)")
                PyPlot.legend(loc="best")
                PyPlot.grid()
                
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
    
        
        
        PyPlot.clf()
        
        subplot(221)
        plot(phi_range[istep_cv_start:end].-phi0, Ib_range[istep_cv_start:end] ,label="bulk")
        plot(phi_range[istep_cv_start:end].-phi0, Ibb_range[istep_cv_start:end] ,label="bulk_grad")
        plot(phi_range[istep_cv_start:end].-phi0, Is_range[istep_cv_start:end] ,label="surf")
        plot(phi_range[istep_cv_start:end].-phi0, r_range[istep_cv_start:end] ,label="reac")
        PyPlot.xlabel("E (V)")
        PyPlot.ylabel("I (A)")
        PyPlot.legend(loc="best")
        PyPlot.grid()
        
        subplot(222)
        if dlcap
            cbl, cs = direct_capacitance(parameters, collect(float(-bound):0.001:float(bound)))
            plot(collect(float(-bound):0.001:float(bound)), (cbl+cs), label="clemens") 
            plot(phi_range[istep_cv_start:end].-phi0, ((Is_range + Ib_range + r_range + Ibb_range)[istep_cv_start:end])/voltrate ,label="rescaled total current")# rescaled by voltrate
        else
            plot(phi_range[istep_cv_start:end].-phi0, ((Is_range + Ib_range + r_range + Ibb_range)[istep_cv_start:end]) ,label="total current")
        end
        PyPlot.xlabel("E (V)")
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
        parn = ["n", "verbose" ,"pyplot", "width", "voltametry", "voltrate", "bound", "sample", "phi0"]
        parv =[n, verbose ,pyplot, width, voltametry, voltrate, bound, sample, @sprintf("%.6g",phi0)]
        for ii in 1:length(parn)
            linestring=string(parn[ii],": ",parv[ii])
            PyPlot.text(0.01+shift, 0.95+height, linestring, fontproperties="monospace")
            height+=-0.05
        end
        #plot(phi_range, r_range ,label="spec1")
        #PyPlot.legend(loc="best")
        #PyPlot.grid()


        if save_files
            out_name=string(
            "A0",@sprintf("%.0f",A0_in),
            "_GA",@sprintf("%.0f",DGA_in),
            "_GR",@sprintf("%.0f",DGR_in),
            "_R0",@sprintf("%.0f",R0_in),
            "_be",@sprintf("%.0f",beta_in),
            "_A",@sprintf("%.0f",A_in)
            )


            CSV.write(string("./data/",out_name,".csv"),out_df)
            if pyplot
                PyPlot.savefig(string("./images/",out_name,".png"))
            end
        end
        #######################################################
        #######################################################
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


# TODO //////////
#   [ ] fix the problem with non-zero current during relaxation
#   [ ] eliminate parameter "n" and "h" used during calculation of dphiB
#   [ ] validate the code to analytic solution with apropriate right-hand-side
