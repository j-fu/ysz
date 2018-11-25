"""
YSZ example cloned from iliq.jl of the TwoPointFluxFVM package.
"""

using Printf
using TwoPointFluxFVM
using PyPlot


mutable struct YSZParameters <:TwoPointFluxFVM.Physics
    TwoPointFluxFVM.@AddPhysicsBaseClassFields
    chi::Float64    # dielectric parameter [1]
    T::Float64      # Temperature [K]
    x_frac::Float64 # Y2O3 mol mixing, x [%] 
    vL::Float64     # volume of one FCC cell, v_L [m^3]
    nu::Float64    # ratio of immobile ions, \nu [1]
    ML::Float64   # averaged molar mass [kg]
    zL::Float64   # average charge number [1]
    DD::Float64   # diffusion coefficient [m^2/s]
    DDs::Float64   # surface adsorption coefficient [m^2/s]
    y0::Float64   # electroneutral value
    dPsi::Float64 # difference of gibbs free energy
    dPsiR::Float64 # difference of gibbs free energy of electrochemical reaction
    areaL::Float64 # volume of one FCC cell, v_L [m^3]
    R0::Float64 # exhange current density [A/m^2]
    pO::Float64 # O2 partial pressure in bar

    e0::Float64  
    eps0::Float64
    kB::Float64  
    N_A::Float64 
    zA::Float64  
    mO::Float64  
    mZr::Float64 
    mY::Float64 
    m_par::Float64
    ms_par::Float64
    
    YSZParameters()= YSZParameters( new())
end

function YSZParameters(this)
    TwoPointFluxFVM.PhysicsBase(this,2)
    this.num_bspecies=[ 1, 0]
    this.chi=1.e1
    this.T=1073
    this.x_frac=0.2
    this.vL=3.35e-29
    this.areaL=(this.vL)^0.6666
    this.nu=0.4
    this.DD=1.0e-13
    this.DDs=1.#e-3   
    this.dPsi=-1.0e-19
    this.dPsiR=1.0#e5
    this.R0=1.0e0
    this.pO=1.0
    #
    this.e0   = 1.602176565e-19  #  [C]
    this.eps0 = 8.85418781762e-12 #  [As/(Vm)] 
    this.kB   = 1.3806488e-23  #  [J/K]  
    this.N_A  = 6.02214129e23  #  [#/mol]
    this.zA  = -2;
    this.mO  = 16/1000/this.N_A  #[kg/#]
    this.mZr = 91.22/1000/this.N_A #  [kg/#]
    this.mY  = 88.91/1000/this.N_A #  [kg/#]
    this.m_par = 2
    this.ms_par = this.m_par
    this.zL  = 4*(1-this.x_frac)/(1+this.x_frac) + 3*2*this.x_frac/(1+this.x_frac) - 2*this.m_par*this.nu
    this.y0  = -this.zL/(this.zA*this.m_par*(1-this.nu))
    this.ML  = (1-this.x_frac)/(1+this.x_frac)*this.mZr + 2*this.x_frac/(1+this.x_frac)*this.mY + this.m_par*this.nu*this.mO
    # this.ML=1.77e-25
    # this.zL=1.8182
    # this.y0=0.9
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


function run_ysz(;n=10, verbose=true,pyplot=false,flux=4, storage=2, xbreaction=2 ,width=1.0e-9)

    h=width/convert(Float64,n)
    X=collect(0.0:h:width)
    
    geom=TwoPointFluxFVM.Graph(X)
    
    parameters=YSZParameters()

    # function flux1!(this::YSZParameters,f,uk,ul)
    #     f[iphi]=this.eps0*(1+this.chi)*(uk[iphi]-ul[iphi])
    #     muk=-log(1-uk[iy])
    #     mul=-log(1-ul[iy])
    #     bp,bm=fbernoulli_pm(2*(uk[iphi]-ul[iphi])+(muk-mul))
    #     f[iy]=bm*uk[iy]-bp*ul[iy]
    # end

    # function flux2!(this::YSZParameters,f,uk,ul)
    #     f[iphi]=this.eps0*(1+this.chi)*(uk[iphi]-ul[iphi])
    #     muk=-log(1-uk[iy])
    #     mul=-log(1-ul[iy])
    #     bp,bm=fbernoulli_pm(-2*(1.0+0.5*(uk[iy]+ul[iy]))*(uk[iphi]-ul[iphi])+(muk-mul))
    #     f[iy]=bm*uk[iy]-bp*ul[iy]
    # end
    # 
    function flux4!(this::YSZParameters,f,uk,ul)
        f[iphi]=this.eps0*(1+this.chi)*(uk[iphi]-ul[iphi])
        muk=log(1-uk[iy])
        mul=log(1-ul[iy])
        bp,bm=fbernoulli_pm(
            -1.0*(ul[iphi]-uk[iphi])*this.zA*this.e0/this.T/this.kB*(
		1.0 + this.mO/this.ML*this.m_par*(1.0-this.nu)*0.5*(uk[iy]+ul[iy])
            )
	    +(mul-muk)*(
		1.0 +this.mO*(1-this.m_par*this.nu)/this.ML
	    )
      	)
        f[iy]=(1+this.mO*this.m_par*(1.0-this.nu)*0.5*(uk[iy]+ul[iy])/this.ML)*this.DD*this.kB/this.mO*(bm*uk[iy]-bp*ul[iy]) # prefactor checked
    end 
    
    # function flux3!(this::YSZParameters,f,uk,ul)
    #     f[iphi]=this.eps0*(1+this.chi)*(uk[iphi]-ul[iphi])
    #     muk=-log(1-uk[iy])
    #     mul=-log(1-ul[iy])
    #     bp,bm=fbernoulli_pm(
    #         -1.0/this.ML/this.kB*(-this.zA*this.e0/this.T*(ul[iphi]-uk[iphi])*(
    #             this.ML + this.mO*this.m_par*(1-.0*this.nu)*0.5*(uk[iy]+ul[iy])
    #         )
    #                               +this.kB*(this.mO*(1-this.m_par*this.nu) + this.ML)*(muk-mul)
    #                               )*this.mO/this.DD/this.kB/this.vL*this.m_par*(1.0-this.nu)*this.mO
    #     )
    #     f[iy]=(1+this.mO*this.m_par*(1.0-this.nu)*0.5*(uk[iy]+ul[iy])/this.ML)*this.DD*this.kB/this.mO*(bm*uk[iy]-bp*ul[iy])*this.vL/this.m_par/(1.0-this.nu)/this.mO/h
    # end 

    # function storage1!(this::YSZParameters, f,u)
    #     f[iphi]=0
	  #     f[iy]=u[iy]
    # end


    function storage2!(this::YSZParameters, f,u)
        f[iphi]=0
        f[iy]=this.mO*this.m_par*(1.0-this.nu)*u[iy]/this.vL
    end

    function reaction!(this::YSZParameters, f,u)
        #f[iphi]=(this.e0/this.vL)*(this.zA*u[iy]*this.m_par*(1-this.nu) + this.zL)
        f[iphi]=-(this.e0/this.vL)*(this.zA*u[iy]*this.m_par*(1-this.nu) + this.zL)
        f[iy]=0
    end
    
    function breaction1!(this::YSZParameters,f,bf,u,bu)
        if  this.bregion==1
            #electroR = this.R0/this.e0*this.mO*sinh(this.dPsiR/this.T/this.kB + 0.5*log(bu[1]) - 0.5*log(1-bu[1]) - 0.25*log(this.pO))
            electroR = 1e-0*((exp(this.dPsiR)*(bu[1]/(1-bu[1]))^0.5*(this.pO)^-0.25 ) - exp(-this.dPsiR)*((bu[1]/(1-bu[1]))^-0.5*(this.pO)^0.25))
            #electroR = 1e-0*((exp(1)*(bu[1]/(1-bu[1]))^0.5*(this.pO)^-0.25 ) - exp(-1)*((bu[1]/(1-bu[1]))^-0.5*(this.pO)^0.25))
            #electroR = -1e-0 
            f[iy]= this.DDs*(
                            this.dPsi + this.kB*this.T/this.mO*(
                              log(u[iy]*(1-bu[1])) - log(bu[1]*(1-u[iy]))
                            )
                  )
            # if bulk chem. pot. > surf. ch.p. then positive flux from bulk to surf
            # sign is negative bcs of the equation implementation
            bf[1]= electroR-this.DDs*( # adsorption term
                            this.dPsi + this.kB*this.T/this.mO*(
                              log(u[iy]*(1-bu[1])) - log(bu[1]*(1-u[iy]))
                            )
                  ) 
            f[iphi]=0
        else
            f[iy]=0        
            f[iphi]=0
        end
    end
    # function breaction3!(this::YSZParameters,f,bf,u,bu)
    #     if  this.bregion==1
    #         f[iy]=this.DDs*(
    #                         this.dPsi + this.kB*this.T*(
    #                           abs(log(u[iy]*(1-bu[1])) - log(bu[1]*(1-u[iy])))
    #                         )
    #               )
    #         bf[1]=-this.DDs*(
    #                         this.dPsi + this.kB*this.T*(
    #                           abs(log(u[iy]*(1-bu[1])) - log(bu[1]*(1-u[iy])))
    #                         )
    #               )
    #         f[iphi]=0
    #     else
    #         f[iy]=0        
    #         f[iphi]=0
    #     end
    # end
    # function breaction2!(this::YSZParameters,f,bf,u,bu)
    #   if  this.bregion==1
    #       f[iy]=(u[iy]-bu[1])
    #       bf[1]=(bu[1]-u[iy])
    #   else
    #       f[1]=0        
    #       f[2]=0
    #   end
    # end
 
    function bstorage!(this::YSZParameters,bf,bu)
        if  this.bregion==1
            bf[1]=this.mO*this.ms_par*(1.0-this.nu)/this.areaL*bu[1]
        else
            bf[1]=0        
        end
    end


    if flux==1 
        fluxx=flux1!
    elseif flux==2 
        fluxx=flux2!
    elseif flux==3 
        fluxx=flux3!
    elseif flux==4 
        fluxx=flux4!
    end

    if storage==1 
        storagex=storage1!
    elseif storage==2
        storagex=storage2!
    end

    if xbreaction == 1
        breaction=breaction1!
    elseif xbreaction == 2
        breaction=breaction2!
    elseif xbreaction == 3
        breaction=breaction3!
    end

    parameters.storage=storagex
    parameters.flux=fluxx
    parameters.reaction=reaction!
    parameters.breaction=breaction
    parameters.bstorage=bstorage!

    printfields(parameters)
    print("weight ", parameters.mO*parameters.m_par*(1.0-parameters.nu)/parameters.vL,"\n")

    sys=TwoPointFluxFVM.System(geom,parameters)

    sys.boundary_values[iphi,1]=1.0e-0
    sys.boundary_values[iphi,2]=0.0e-3
    
    sys.boundary_factors[iphi,1]=TwoPointFluxFVM.Dirichlet
    sys.boundary_factors[iphi,2]=TwoPointFluxFVM.Dirichlet

    sys.boundary_values[iy,2]=parameters.y0
    sys.boundary_factors[iy,2]=TwoPointFluxFVM.Dirichlet
    
    inival=unknowns(sys)
    inival.=0.0
    
    inival_bulk=bulk_unknowns(sys,inival)
    for inode=1:size(inival_bulk,2)
        #inival_bulk[iphi,inode]=0.0e-3
        inival_bulk[iy,inode]= parameters.y0
    end
    inival_boundary = boundary_unknowns(sys,inival,1)
    inival_boundary[1]= parameters.y0
    if false 
      print("phi_init: ") 
      print(inival_bulk[iphi,:])
      print("\n")
      print("y_init: ") 
      print(inival_bulk[iy,:])
      print("\n")
      print("ys_init: ") 
      print(inival_boundary)
      print(parameters.areaL)
    end 

    #parameters.eps=1.0e-2
    #parameters.a=5
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
    time_range=zeros(0)
    istep=0
    Ub=zeros(0)
    tend=1.0e-4
    tstep=1.0e-7
    append!(time_range,time)
    append!(Ub,inival_boundary[1])
    while time<tend
        time=time+tstep
        U=solve(sys,inival,control=control,tstep=tstep)
        inival.=U
        # for i=1:length(inival)
        #     inival[i]=U[i]
        # end
        if verbose
            @printf("time=%g\n",time)
        end
        U_bulk=bulk_unknowns(sys,U)
        U_bound=boundary_unknowns(sys,U,1)
        append!(time_range,time)
        append!(Ub,U_bound[1,1])

        if pyplot && istep%10 == 0
        #if pyplot 
            @printf("max1=%g max2=%g maxb=%g\n",maximum(U_bulk[1,:]),maximum(U_bulk[2,:]),maximum(U_bound))
            PyPlot.clf()
            subplot(211)
            plot(X,U_bulk[1,:],label="spec1")
            plot(X,U_bulk[2,:],label="spec2")
            PyPlot.legend(loc="best")
            PyPlot.grid()
            subplot(212)
            plot(time_range,Ub,label="U_b")
            PyPlot.legend(loc="best")
            PyPlot.grid()
            pause(1.0e-10)
            print(U_bound)
        end

        # if pyplot
        #     PyPlot.clf()
        #     PyPlot.plot(geom.node_coordinates[1,:],U_bulk[iphi,:], label="Potential", color="g")
        #     PyPlot.plot(geom.node_coordinates[1,:],U_bulk[iy,:], label="y", color="b")
        #     PyPlot.grid()
        #     PyPlot.legend(loc="upper right")
        #     PyPlot.pause(1.0e-10)
        # end
        tstep*=1.05
    end
end



if !isinteractive()
    @time run_ysz(n=100,pyplot=true)
    waitforbuttonpress()
end
