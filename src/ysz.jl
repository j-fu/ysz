"""
YSZ example cloned from iliq.jl of the TwoPointFluxFVM package.
"""

using Printf
using TwoPointFluxFVM

if !isinteractive()
    using PyPlot
end


mutable struct YSZParameters <:TwoPointFluxFVM.Physics
    TwoPointFluxFVM.@AddPhysicsBaseClassFields
    chi::Float64    # dielectric parameter [1]
    T::Float64      # Temperature [K]
    x_frac::Float64	# Y2O3 mol mixing, x [%] 
    vL::Float64	    # volume of one FCC cell, v_L [m^3]
    nu::Float64		  # ratio of immobile ions, \nu [1]
    ML::Float64			# averaged molar mass [kg]
    zL::Float64			# average charge number [1]
    DD::Float64			# diffusion coefficient [m^2/s]
    y0::Float64			# electroneutral value
    YSZParameters() = YSZParameters(new())
end

function YSZParameters(this::YSZParameters)
    TwoPointFluxFVM.PhysicsBase(this,2)
    this.chi=1e1
    this.T=1073
    this.x_frac=0.1
    this.vL=3.35e-29
    this.nu=0.1
    this.ML=1.77e-25
    this.zL=1.8182
    this.DD=1.0e-9
    this.y0=0.9
    return this
end

function setDependentYSZ(this::YSZParameters)
    this.zL		= 4*(1-this.x_frac)/(1+this.x_frac) + 3*2*this.x_frac/(1+this.x_frac) - 2*m_par*this.nu
    this.y0  = -this.zL/(zA*m_par*(1-this.nu))
    this.ML  = (1-this.x_frac)/(1+this.x_frac)*mZr + 2*this.x_frac/(1+this.x_frac)*mY + m_par*this.nu*mO
    if true
	print(
            "Parameters\n---------------------------------\n",
	    " chi =",this.chi,
	    "\n T =", this.T,
	    "\n x_frac=", this.x_frac,# Y2O3 mol mixing, x [%] 
	    "\n lattice_volume=", this.vL,# volume of one FCC cell, v_L [m^3]
	    "\n nu=", this.nu,# ratio of immobile ions, \nu [1]
	    "\n ML=", this.ML,# averaged molar mass [kg]
	    "\n zL=", this.zL,# average charge number [1]
	    "\n DD=", this.DD,# diffusion coefficient [m^2/s]
	    "\n y0=", this.y0,#
            "\n-------------------------------------------\n",
	)
    end
end

function printCoefficients(this::YSZParameters)
    print(
	"epsilon		:",	eps0*(1+this.chi),"\n",
	"phi coef		:",1.0/this.ML/kB*(-zA*e0/this.T*this.ML*mO/this.DD/kB/this.vL*m_par*(1.0-this.nu))*mO,"\n",
	"y coef			:",1.0/this.ML/kB*kB*(mO*(1-this.nu) + this.ML)*mO/this.DD/kB/this.vL*m_par*(1.0-this.nu)*mO,"\n",
    )
end

const e0   = 1.602176565e-19   #  [C]
const eps0 = 8.85418781762e-12 #  [As/(Vm)] 
const kB   = 1.3806488e-23     #  [J/K]  
const N_A   = 6.02214129e23    #  [#/mol]
const zA  = -2;

const mO  = 16/1000/N_A		#		[kg/#]
const mZr = 91.22/1000/N_A#		[kg/#]
const mY  = 88.91/1000/N_A#		[kg/#]

const m_par   = 2

const iphi=1
const ic=2
const beps=1.0e-4

function run_ysz(;n=100,pyplot=false )

    h=1.0/convert(Float64,n)
    geom=TwoPointFluxFVM.Graph(collect(0:h:1))
    
    parameters=YSZParameters()
    setDependentYSZ(parameters)
    printCoefficients(parameters)

    function flux!(this::YSZParameters,f,uk,ul)
	f[iphi]=eps0*(1+this.chi)*(uk[iphi]-ul[iphi])
        muk=-log(1-uk[ic])
        mul=-log(1-ul[ic])
        #        bp,bm=fbernoulli_pm(2*(uk[iphi]-ul[iphi])+(muk-mul))
        bp,bm=fbernoulli_pm(
	    1.0/this.ML/kB*(
                -zA*e0/this.T*(
                    ul[iphi]-uk[iphi]
                )*(
		    this.ML + mO*m_par*(1-.0*this.nu)*0.5*(uk[ic]+ul[ic])
                )
		+kB*(mO*(1-m_par*this.nu) + this.ML)*(muk-mul)
	    )*mO/this.DD/kB/this.vL*m_par*(1.0-this.nu)*mO
	)
	#print(bm,"		", bp)
	f[ic]=this.DD*kB/mO*(bm*uk[ic]-bp*ul[ic])*this.vL/m_par/(1.0-this.nu)/mO
    end 


    function storage!(this::YSZParameters, f,u)
        f[iphi]=0
        f[ic]=u[ic]
    end
    
    function reaction!(this::YSZParameters, f,u)
	f[iphi]=e0/this.vL*(zA*u[ic]*m_par*(1-this.nu) + this.zL)
        f[ic]=0
    end
    
    parameters.storage=storage!
    parameters.flux=flux!
    parameters.reaction=reaction!
    
    sys=TwoPointFluxFVM.System(geom,parameters)
    sys.boundary_values[iphi,1]=1.0e-1
    sys.boundary_values[iphi,2]=0.0e-3
    
    sys.boundary_factors[iphi,1]=TwoPointFluxFVM.Dirichlet
    sys.boundary_factors[iphi,2]=TwoPointFluxFVM.Dirichlet

    sys.boundary_values[ic,2]=parameters.y0
    sys.boundary_factors[ic,2]=TwoPointFluxFVM.Dirichlet
    
    inival=unknowns(sys)

    inival_bulk=bulk_unknowns(sys,inival)
    for inode=1:size(inival_bulk,2)
        inival_bulk[iphi,inode]=0.0e-3
        inival_bulk[ic,inode]= parameters.y0
    end

    #parameters.eps=1.0e-2
    #parameters.a=5
    control=TwoPointFluxFVM.NewtonControl()
    control.verbose=true
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
            plot(geom.node_coordinates[1,:],U_bulk[iphi,:])
            plot(geom.node_coordinates[1,:],U_bulk[ic,:])
            pause(1.0e-10)
        end
        tstep*=1.2
    end
end



if !isinteractive()
    @time run_ysz(n=100,pyplot=true)
    waitforbuttonpress()
end

