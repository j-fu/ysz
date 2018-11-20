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
    y0::Float64   # electroneutral value
    
    e0::Float64  
    eps0::Float64
    kB::Float64  
    N_A::Float64 
    zA::Float64  
    mO::Float64  
    mZr::Float64 
    mY::Float64 
    m_par::Float64
    
    YSZParameters()= YSZParameters( new())
end

function YSZParameters(this)
    TwoPointFluxFVM.PhysicsBase(this,2)
    this.chi=1.e1
    this.T=1073
    this.x_frac=0.2
    this.vL=3.35e-29
    this.nu=0.4
    this.DD=1.0e-13
    this.e0   = 1.602176565e-19  #  [C]
    this.eps0 = 8.85418781762e-12 #  [As/(Vm)] 
    this.kB   = 1.3806488e-23  #  [J/K]  
    this.N_A   = 6.02214129e23  #  [#/mol]
    this.zA  = -2;
    this.mO  = 16/1000/this.N_A  #[kg/#]
    this.mZr = 91.22/1000/this.N_A #  [kg/#]
    this.mY  = 88.91/1000/this.N_A #  [kg/#]
    this.m_par = 2
    this.zL  = 4*(1-this.x_frac)/(1+this.x_frac) + 3*2*this.x_frac/(1+this.x_frac) - 2*this.m_par*this.nu
    this.y0  = -this.zL/(this.zA*this.m_par*(1-this.nu))
    this.ML  = (1-this.x_frac)/(1+this.x_frac)*this.mZr + 2*this.x_frac/(1+this.x_frac)*this.mY + this.m_par*this.nu*this.mO
    #       this.ML=1.77e-25
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


function run_ysz(;n=1000,pyplot=false,flux=4, storage=2, width=1.0e-9)

    h=width/convert(Float64,n)
    geom=TwoPointFluxFVM.Graph(collect(0:h:width))
    
    parameters=YSZParameters()


    function flux1!(this::YSZParameters,f,uk,ul)
        f[iphi]=this.eps0*(1+this.chi)*(uk[iphi]-ul[iphi])
        muk=-log(1-uk[iy])
        mul=-log(1-ul[iy])
        bp,bm=fbernoulli_pm(2*(uk[iphi]-ul[iphi])+(muk-mul))
        f[iy]=bm*uk[iy]-bp*ul[iy]
    end

    function flux2!(this::YSZParameters,f,uk,ul)
        f[iphi]=this.eps0*(1+this.chi)*(uk[iphi]-ul[iphi])
        muk=-log(1-uk[iy])
        mul=-log(1-ul[iy])
        bp,bm=fbernoulli_pm(-2*(1.0+0.5*(uk[iy]+ul[iy]))*(uk[iphi]-ul[iphi])+(muk-mul))
        f[iy]=bm*uk[iy]-bp*ul[iy]
    end
    
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
        f[iy]=this.DD*this.kB/this.mO*(bm*uk[iy]-bp*ul[iy])
    end 
    
    function flux3!(this::YSZParameters,f,uk,ul)
        f[iphi]=this.eps0*(1+this.chi)*(uk[iphi]-ul[iphi])
        muk=-log(1-uk[iy])
        mul=-log(1-ul[iy])
        bp,bm=fbernoulli_pm(
            -1.0/this.ML/this.kB*(-this.zA*this.e0/this.T*(ul[iphi]-uk[iphi])*(
                this.ML + this.mO*this.m_par*(1-.0*this.nu)*0.5*(uk[iy]+ul[iy])
            )
                                  +this.kB*(this.mO*(1-this.m_par*this.nu) + this.ML)*(muk-mul)
                                  )*this.mO/this.DD/this.kB/this.vL*this.m_par*(1.0-this.nu)*this.mO
        )
        f[iy]=this.DD*this.kB/this.mO*(bm*uk[iy]-bp*ul[iy])*this.vL/this.m_par/(1.0-this.nu)/this.mO/h
    end 

    function storage1!(this::YSZParameters, f,u)
        f[iphi]=0
	f[iy]=u[iy]
    end


    function storage2!(this::YSZParameters, f,u)
        f[iphi]=0
	f[iy]=this.mO*this.m_par*(1.0-this.nu)*u[iy]/this.vL
    end

    function reaction!(this::YSZParameters, f,u)
        #f[iphi]=(this.e0/this.vL)*(this.zA*u[iy]*this.m_par*(1-this.nu) + this.zL)
        f[iphi]=-(this.e0/this.vL)*(this.zA*u[iy]*this.m_par*(1-this.nu) + this.zL)
        f[iy]=0
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

    parameters.storage=storagex
    parameters.flux=fluxx
    parameters.reaction=reaction!

    printfields(parameters)


    sys=TwoPointFluxFVM.System(geom,parameters)

    sys.boundary_values[iphi,1]=5.0e-1
    sys.boundary_values[iphi,2]=0.0e-3
    
    sys.boundary_factors[iphi,1]=TwoPointFluxFVM.Dirichlet
    sys.boundary_factors[iphi,2]=TwoPointFluxFVM.Dirichlet

    sys.boundary_values[iy,2]=parameters.y0
    sys.boundary_factors[iy,2]=TwoPointFluxFVM.Dirichlet
    
    inival=unknowns(sys)
    
    inival_bulk=bulk_unknowns(sys,inival)
    for inode=1:size(inival_bulk,2)
        inival_bulk[iphi,inode]=0.0e-3
        inival_bulk[iy,inode]= parameters.y0
    end
    #parameters.eps=1.0e-2
    #parameters.a=5
    control=TwoPointFluxFVM.NewtonControl()
    control.verbose=true
    control.damp_initial=0.01
    control.damp_growth=2
    t=0.0
    tend=1.0
    tstep=1.0e-10
    while t<tend
        t=t+tstep
        U=solve(sys,inival,control=control,tstep=tstep)
        for i=1:length(inival)
            inival[i]=U[i]
        end

        @printf("time=%g\n",t)
        U_bulk=bulk_unknowns(sys,U)
        if pyplot
            PyPlot.clf()
            PyPlot.plot(geom.node_coordinates[1,:],U_bulk[iphi,:], label="Potential", color="g")
            PyPlot.plot(geom.node_coordinates[1,:],U_bulk[iy,:], label="y", color="b")
            PyPlot.grid()
            PyPlot.legend(loc="upper right")
            PyPlot.pause(1.0e-10)
        end
        tstep*=1.2
    end
end



if !isinteractive()
    @time run_ysz(n=100,pyplot=true)
    waitforbuttonpress()
end
