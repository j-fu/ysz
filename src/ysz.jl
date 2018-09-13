"""
YSZ example cloned from iliq.jl of the TwoPointFluxFVM package.
"""

using Printf
using TwoPointFluxFVM

if !isinteractive()
    using PyPlot
end


mutable struct YSZParameters <:FVMParameters
    number_of_species::Int64
    chi::Float64    # dielectric parameter [1]
    T::Float64      # Temperature [K]
    x_frac::Float64	# Y2O3 mol mixing, x [%] 
    vL::Float64	    # volume of one FCC cell, v_L [m^3]
    nu::Float64		  # ratio of immobile ions, \nu [1]
    ML::Float64			# averaged molar mass [kg]
    zL::Float64			# average charge number [1]
    DD::Float64			# diffusion coefficient [m^2/s]
		y0::Float64			# electroneutral value
    function YSZParameters()
        new(2, 1e1, 1073, 0.1, 3.35e-29, 0.1, 1.77e-25, 1.8182, 1.0e-9, 0.9)
		end
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
    geom=FVMGraph(collect(0:h:1))
    
    parameters=YSZParameters()
		setDependentYSZ(parameters)
    function flux!(this::YSZParameters,f,uk,ul)
				f[iphi]=eps0*(1+this.chi)*(uk[iphi]-ul[iphi])
        muk=-log(1-uk[ic])
        mul=-log(1-ul[ic])
#        bp,bm=fbernoulli_pm(2*(uk[iphi]-ul[iphi])+(muk-mul))
        bp,bm=fbernoulli_pm(
                  (
                   ul[iphi]-uk[iphi]
                  )*(
                     1.0 + uk[ic]+ul[ic]
                    )
                  +2.0*(muk-mul))
        f[ic]=bm*uk[ic]-bp*ul[ic]
    end 


    function classflux!(this::YSZParameters,f,uk,ul)
        f[iphi]=this.eps*(uk[iphi]-ul[iphi])
        arg=uk[iphi]-ul[iphi]
        bp,bm=fbernoulli_pm(uk[iphi]-ul[iphi])
        f[ic]=bm*uk[ic]-bp*ul[ic]
    end 

    function storage!(this::FVMParameters, f,u)
        f[iphi]=0
        f[ic]=u[ic]
    end
    
    function reaction!(this::FVMParameters, f,u)
			f[iphi]=e0*(zA*u[ic]*m_par*(1-this.nu) + this.zL)/this.vL
        f[ic]=0
    end
    
    
    sys=TwoPointFluxFVMSystem(geom,parameters=parameters, 
                              storage=storage!, 
                              flux=flux!, 
                              reaction=reaction!
                              )
    sys.boundary_values[iphi,1]=1
    sys.boundary_values[iphi,2]=0.0
    
    sys.boundary_factors[iphi,1]=Dirichlet
    sys.boundary_factors[iphi,2]=Dirichlet

		sys.boundary_values[ic,2]=parameters.y0
    sys.boundary_factors[ic,2]=Dirichlet
    
    inival=unknowns(sys)
    for inode=1:size(inival,2)
        inival[iphi,inode]=0
        inival[ic,inode]=parameters.y0
    end
    #parameters.eps=1.0e-2
    #parameters.a=5
    control=FVMNewtonControl()
    control.verbose=true
    t=0.0
    tend=1.0
    tstep=1.0e-10
    while t<tend
        t=t+tstep
        U=solve(sys,inival,control=control,tstep=tstep)
        for i=1:size(inival,2)
            inival[iphi,i]=U[iphi,i]
            inival[ic,i]=U[ic,i]
        end
        @printf("time=%g\n",t)
        if pyplot
            PyPlot.clf()
            plot(geom.Nodes[1,:],U[iphi,:])
            plot(geom.Nodes[1,:],U[ic,:])
            pause(1.0e-10)
        end
        tstep*=1.2
    end
end



if !isinteractive()
    @time run_ysz(n=100,pyplot=true)
    waitforbuttonpress()
end
