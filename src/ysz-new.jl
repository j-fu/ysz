module YSZNew

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
            this.mO*this.A0*
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
            - this.mO*electroR - this.mO*this.A0*
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
















function run_new(;test=false, print_bool=false, verbose=false ,pyplot=false, save_files=false, width=10.0e-9,  dx_exp=-9, voltammetry=false, dlcap=false, voltrate=0.005, upp_bound=0.5, low_bound=-0.5, sample=40, prms_in=[21.71975544711280, 20.606423236896422, 0.0905748, -0.708014, 0.6074566741435283, 0.1], nu_in=0.9)

    # prms_in = [ A0, R0, DGA, DGR, beta, A ]

    # Geometry of the problem
    AreaEllyt = 0.000201 * 0.6      # m^2   (geometrical area)*(1 - porosity)
    width_Ellyt = 0.00045           # m     width of the half-cell
    if dlcap
        AreaEllyt = 1.0      # m^2    
        if print_bool
            println("dlcap > area = 1")
        end
    end
    #
    dx_start = 10^convert(Float64,dx_exp)
    X=width*TwoPointFluxFVM.geomspace(0.0,1.0,dx_start,1e-1)
    #println("X = ",X)
    
    #
    geom=TwoPointFluxFVM.Graph(X)
    #
    
    parameters=YSZParameters()
    # for a parametric study
    eV = parameters.e0   # electronvolt [J] = charge of electron * 1[V]
    parameters.A0 = 10.0^prms_in[1]      # [1 / s]
    parameters.R0 = 10.0^prms_in[2]      # [1 / m^2 s]
    if dlcap
        parameters.R0 = 0
        if print_bool
            println("dlcap > R0= ",parameters.R0)
        end
    end
    parameters.DGA = prms_in[3] * eV    # [J]
    parameters.DGR = prms_in[4] * eV    # [J]
    parameters.beta = prms_in[5]       # [1]
    parameters.A = 10.0^prms_in[6]        # [1]
    

    parameters.nu = nu_in
    
    # update the "computed" values in parameters
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
        inival_bulk[iphi,inode]=0.0
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
    
    # code for performing the CV
    if voltammetry
        istep=1
        phi=0
    #        phi=phi0

        # inicializing storage lists
        y0_range=zeros(0)
        ys_range=zeros(0)
        phi_range=zeros(0)
        #
        Is_range=zeros(0)
        Ib_range=zeros(0)
        Ibb_range=zeros(0)
        Ir_range=zeros(0)
        
        if save_files
            out_df = DataFrame(t = Float64[], U = Float64[], Itot = Float64[], Ibu = Float64[], Isu = Float64[], Ire = Float64[])
        end
        
        cv_cycles = 1
        relaxation_length = 1    # how many "(cv_cycle/4)" should relaxation last
        relax_counter = 0
        istep_cv_start = -1
        time_range = zeros(0)  # [s]

        if print_bool
            print("calculating linear potential sweep\n")
        end
        direction_switch_count = 0

        tstep=((upp_bound-low_bound)/2)/voltrate/sample   
        if print_bool
            @printf("tstep %g = \n", tstep)
        end
        if phi0 > 0
            dir=1
        else
            dir=-1
        end
        
        if pyplot
            PyPlot.close()
            PyPlot.ion()
            PyPlot.figure(figsize=(10,8))
            #PyPlot.figure(figsize=(5,5))
        end
        
        state = "ramp"
        if print_bool
            println("phi_equilibrium = ",phi0)
            println("ramp ......")
        end
        
        if test
            U = inival
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
            
                direction_switch_count +=1
                if print_bool
                    print(direction_switch_count,", ")
                end
            end            
            if state=="cv_is_on" && (dir > 0) && (phi > phi0 + 0.000001) && (direction_switch_count >=2*cv_cycles)
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

            # extracting the computed data
            U_bulk=bulk_unknowns(sys,U)
            U_bound=boundary_unknowns(sys,U,1)
            
            
            # storing data
            append!(y0_range,U_bulk[iy,1])
            append!(ys_range,U_bound[1,1])
            append!(phi_range,phi)
            #
            append!(Ib_range,Ib)
            append!(Is_range,Is)
            append!(Ibb_range,Ibb)
            append!(Ir_range, Ir)
            #
            append!(time_range,tstep*istep)
            
            if save_files
                if state=="cv_is_on"
                    if dlcap
                        push!(out_df,[(istep-istep_cv_start)*tstep   phi-phi0    (Ib+Is+Ir)/voltrate    Ib/voltrate    Is/voltrate    Ir/voltrate])
                    else
                        push!(out_df,[(istep-istep_cv_start)*tstep   phi-phi0    Ib+Is+Ir    Ib    Is    Ir])
                    end
                end
            end
            
            
            
            # plotting                  


            if pyplot && istep%10 == 0

                num_subplots=4
                ys_marker_size=4
                PyPlot.subplots_adjust(hspace=0.5)
            
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
                    PyPlot.ylim(-0.5,1.1)
                    PyPlot.xlabel("x (mm)")
                    PyPlot.legend(loc="best")
                    PyPlot.grid()
                end
                
                
                if num_subplots > 2
                    subplot(num_subplots*100 + 13)
                    plot(time_range,phi_range,label="phi_S (V)")
                    plot(time_range,y0_range,label="y(0)")
                    plot(time_range,ys_range,label="y_s")
                    PyPlot.xlabel("t (s)")
                    PyPlot.legend(loc="best")
                    PyPlot.grid()
                end
                
                if num_subplots > 3
                    subplot(num_subplots*100 + 14)
                    plot(time_range,Is_range + Ib_range + Ir_range + Ibb_range,label="total I (A)")
                    PyPlot.xlabel("t (s)")
                    PyPlot.legend(loc="best")
                    PyPlot.grid()
                end
                                
                pause(1.0e-10)
            end
            
            # preparing for the next step
            istep+=1
            if state=="relaxation"
                relax_counter += 1
                #println("relaxation ... ",relax_counter/sample*100,"%")
            else
                phi+=voltrate*dir*tstep
            end
        end
        
        
        
        # the finall plot
        if pyplot
            PyPlot.pause(5)
            
            PyPlot.clf()
            #PyPlot.close()
            #PyPlot.figure(figsize=(5,5))
            
            cv_range = (istep_cv_start+1):length(phi_range)


            subplot(221)
            if dlcap
                plot(phi_range[cv_range].-phi0,( Ib_range[cv_range] )/voltrate,"blue", label="bulk")
                #plot(phi_range[cv_range].-phi0,( Ibb_range[cv_range])/voltrate,label="bulk_grad")
                plot(phi_range[cv_range].-phi0,( Is_range[cv_range] )/voltrate,"green", label="surf")
                plot(phi_range[cv_range].-phi0,( Ir_range[cv_range]  )/voltrate,"red", label="reac")
            else
                plot(phi_range[cv_range].-phi0, Ib_range[cv_range] ,"blue", label="bulk")
                #plot(phi_range[cv_range].-phi0, Ibb_range[cv_range] ,label="bulk_grad")
                plot(phi_range[cv_range].-phi0, Is_range[cv_range] ,"green",label="surf")
                plot(phi_range[cv_range].-phi0, Ir_range[cv_range] ,"red",label="reac")
            end
            if dlcap
                PyPlot.xlabel("nu (V)")
                PyPlot.ylabel(L"Capacitance (F/m$^2$)")  
                PyPlot.legend(loc="best")
                PyPlot.xlim(-0.5, 0.5)
                PyPlot.ylim(0, 5)
                PyPlot.grid()
                PyPlot.show()
                PyPlot.pause(5)
                
                PyPlot.clf()
                plot(phi_range[cv_range].-phi0,( (Ib_range+Is_range+Ir_range)[cv_range]  )/voltrate,"brown", label="total")
                PyPlot.xlabel("nu (V)")
                PyPlot.ylabel(L"Capacitance (F/m$^2$)") 
                PyPlot.legend(loc="best")
                PyPlot.xlim(-0.5, 0.5)
                PyPlot.ylim(0, 5)
                PyPlot.grid()
                PyPlot.show()
                #PyPlot.pause(10)
            else
                PyPlot.xlabel(L"\eta \ (V)")
                PyPlot.ylabel(L"I \ (A)")
                PyPlot.legend(loc="best")
                PyPlot.grid()
            end
            
            
            subplot(222)
            if dlcap
                cbl, cs = direct_capacitance(parameters, collect(float(low_bound):0.001:float(upp_bound)))
                plot(collect(float(low_bound):0.001:float(upp_bound)), (cbl+cs), label="tot CG") 
                plot(collect(float(low_bound):0.001:float(upp_bound)), (cbl), label="b CG") 
                plot(collect(float(low_bound):0.001:float(upp_bound)), (cs), label="s CG") 
                # rescaled by voltrate
                plot(phi_range[cv_range].-phi0, ((Is_range + Ib_range + Ir_range + Ibb_range)[cv_range])/voltrate ,label="rescaled total current")
            else
                plot(phi_range[cv_range].-phi0, ((Is_range + Ib_range + Ir_range + Ibb_range)[cv_range]) ,label="total current")
            end
            PyPlot.xlabel(L"\eta \ (V)")
            PyPlot.ylabel(L"I \ (A)")
            PyPlot.legend(loc="best")
            PyPlot.grid()
            
            
            subplot(223)
            #plot(phi_range, Ir_range ,label="spec1")
            plot(time_range,phi_range,label="phi_s (V)")        
            plot(time_range,y0_range,label="y(0)")
            plot(time_range,ys_range,label="y_s")
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
            parn = ["verbose" ,"pyplot", "width", "voltammetry", "voltrate", "low_bound", "upp_bound", "sample", "phi0"]
            parv =[verbose ,pyplot, width, voltammetry, voltrate, low_bound, upp_bound, sample, @sprintf("%.6g",phi0)]
            for ii in 1:length(parn)
                    linestring=string(parn[ii],": ",parv[ii])
                    PyPlot.text(0.01+shift, 0.95+height, linestring, fontproperties="monospace")
                    height+=-0.05
            end
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
        if test
            I1 = integrate(sys, reaction!, U)
            #println(I1)
            return I1[1]
        end
    end
end


end
