module ysz_nih2h2o

using Printf
using VoronoiFVM
using Plots
using PyPlot
using DataFrames
using CSV
using LeastSquaresOptim
using Pkg
##########################################
# internal import of YSZ repo ############
cur_dir=pwd()
cd("../src")
src_dir=pwd()
cd(cur_dir)

push!(LOAD_PATH,src_dir)
include("../prototypes/timedomain_impedance.jl")
using ysz_model_fitted_parms
using ysz_model_Ni
const label_ysz_model = ysz_model_fitted_parms
const label_Ni_model = ysz_model_Ni

# --------- end of YSZ import ---------- #
##########################################

function run_new(;testplot=true, print_bool=false, verbose=false, pyplot=false, save_files=false, width=10.0e-9,  dx_exp=-9, EIS_TDS=false, EIS_IS=false, nu_in=0.9, tref=0, EIS=false)

    dx_start = 10^convert(Float64,dx_exp)
    X=width*VoronoiFVM.geomspace(0.0,1.0,dx_start,1e-1)
    
    #
    grid=VoronoiFVM.Grid(X)
    #
    parameters=label_Ni_model.YSZParameters()
    
    # for a parametric study
    parameters.nu = nu_in
    
    # update the "computed" values in parameters
    parameters = label_Ni_model.YSZParameters_update(parameters)

    physics=VoronoiFVM.Physics(
        data=parameters,
        num_species=parameters.num_active,
        storage=label_ysz_model.storage!,
        flux=label_ysz_model.flux!,
        reaction=label_ysz_model.reaction!,
        breaction=label_Ni_model.breactionNi!,
        bstorage=label_Ni_model.bstorageNi!
    )
    #
    if print_bool
        label_ysz_model.printfields(parameters)
    end

    sys=VoronoiFVM.SparseSystem(grid,physics)
    #sys=VoronoiFVM.DenseSystem(grid,physics)
    enable_species!(sys,iphi,[1])
    enable_species!(sys,iy,[1])
    for ii=3:parameters.num_active
      enable_boundary_species!(sys,ii,[1])
    end
    #
    sys.boundary_values[iphi,1]=-1.0e-1
    sys.boundary_values[iphi,2]=0.0e-3
    #
    sys.boundary_factors[iphi,1]=VoronoiFVM.Dirichlet
    sys.boundary_factors[iphi,2]=VoronoiFVM.Dirichlet
    #
    sys.boundary_values[iy,2]=parameters.y0
    sys.boundary_factors[iy,2]=VoronoiFVM.Dirichlet
    #
    inival=unknowns(sys)
    inival.=0.0
    #
    
    # phi0 = label_ysz_model.equil_phi(parameters)
    if print_bool
        println("phi0 = ",phi0)
    end
    
    @views inival[iphi,:].=.00
    @views inival[iy,:].=parameters.y0
    @views inival[3,1]=0.1
    @views inival[4,1]=0.9
    @views inival[5,1]=1.0
    @views inival[6:parameters.num_active, 1] .= 1.0e-1/convert(Float64,parameters.num_active+ 1 )

    control=VoronoiFVM.NewtonControl()
    control.verbose=verbose
    control.tol_linear=1.0e-4
    control.tol_relative=1.0e-4
    #control.tol_absolute=1.0e-4
    #control.max_iterations=3
    control.max_lureuse=0
    control.damp_initial=1.0e-3
    control.damp_growth=1.9
    time=0.0
    
    if testplot
        U = unknowns(sys)
        T= 1e-6*VoronoiFVM.geomspace(0.0, 1.0, 1e-2, 1e-1)
        Tsteps = [T[ii] - T[ii-1] for ii=2:length(T)]
        #
        solStor= Array{Float64,3}(undef, length(T), size(inival)[1], size(inival)[2])
        solStor[1,:,:] = inival
        for (ii, tstep) in enumerate(Tsteps)
            solve!(U,inival,sys,control=control,tstep=tstep)
            inival.=U
            solStor[ii+1,:,:] = U
            if verbose
                @printf("time=%g\n",t)
            end
        end
        p1=Plots.heatmap(sys.grid.coord[1,:],T, solStor[:,1,:])
        p2=Plots.heatmap(sys.grid.coord[1,:],T, solStor[:,2,:])
        p3=Plots.plot(grid=true)
        [Plots.plot!(p3, T, abs.(solStor[:, ii,1]),ylabel="U_b",xlabel="t", yaxis=:log10) for ii=6:parameters.num_active]
        p=Plots.plot(p1,p2,p3, layout=(3,1),legend=true, dpi=150)
        gui(p)
    end

		#
		# impedance spectroscopy
		#
    if EIS
    # Calculate steady state solution
    steadystate = unknowns(sys)
        
    phi_OCV = 0.0374762 # needs to be adjusted 
    
    excited_spec=iphi
    excited_bc=1
    excited_bcval=phi_OCV
    
    # relaxation
    sys.boundary_values[iphi,1]= phi_OCV

    # direct steady state solution is not obtainable
   	T= 1e-6*VoronoiFVM.geomspace(0.0, 1.0, 1e-2, 1e-1)
    Tsteps = [T[ii] - T[ii-1] for ii=2:length(T)]
		for (ii, tstep) in enumerate(Tsteps) 
            solve!(steadystate,inival,sys,control=control,tstep=tstep)
            inival.=steadystate
		end

    # Create impedance system
    isys=VoronoiFVM.ImpedanceSystem(sys,steadystate,excited_spec, excited_bc)

    function meas_stdy(meas, u) 
        return label_Ni_model.currentNi_stdy!(meas, u, sys)
    end

    function meas_tran(meas, u) 
        return label_Ni_model.currentNi_tran!(meas, u, sys)
    end

    # Derivatives of measurement functionals
    # For the Julia magic behind this we need the measurement functionals
    # as mutating functions writing on vectors.
    dmeas_stdy=measurement_derivative(sys,meas_stdy,steadystate)
    dmeas_tran=measurement_derivative(sys,meas_tran,steadystate)


    
    # Impedance arrays
    z_timedomain=zeros(Complex{Float64},0)
    z_freqdomain=zeros(Complex{Float64},0)
    all_w=zeros(0)

    w0 = 1.0e-6
    w1 = 1.0e6
    
    w = w0

    # Frequency loop
    @time while w<w1
        @show w
        push!(all_w,w)
        if EIS_IS
            # Here, we use the derivatives of the measurement functional
            zfreq=freqdomain_impedance(isys,w,steadystate,excited_spec,excited_bc,excited_bcval, dmeas_stdy, dmeas_tran)
            push!(z_freqdomain,1.0/zfreq)
            @show zfreq
        end
        
        if EIS_TDS
            # Similar API, but use the the measurement functional themselves
            ztime=timedomain_impedance(sys,w,steadystate,excited_spec,excited_bc,excited_bcval,meas_stdy, meas_tran,
                                   tref=tref,
                                   fit=true)
            push!(z_timedomain,1.0/ztime)
            @show ztime
        end

        
        # growth factor such that there are 10 points in every order of magnitude
        # (which is consistent with "freq" list below)
        w=w*1.25892
    end


    if pyplot
        function positive_angle(z)
            ϕ=angle(z)
            if ϕ<0.0
                ϕ=ϕ+2*π
            end
            return ϕ
        end

        PyPlot.clf()
        PyPlot.subplot(311)
        PyPlot.grid()

        if EIS_IS
            PyPlot.semilogx(all_w,positive_angle.(1.0/z_freqdomain)',label="\$i\\omega\$",color=:red)
        end
        if EIS_TDS
            PyPlot.semilogx(all_w,positive_angle.(1.0/z_timedomain)',label="\$\\frac{d}{dt}\$",color=:green)
        end
        PyPlot.xlabel("\$\\omega\$")
        PyPlot.ylabel("\$\\phi\$")
        PyPlot.legend(loc="upper left")


        PyPlot.subplot(312)
        PyPlot.grid()
        if EIS_IS
            PyPlot.loglog(all_w,abs.(1.0/z_freqdomain)',label="\$i\\omega\$",color=:red)
        end
        if EIS_TDS
            PyPlot.loglog(all_w,abs.(1.0/z_timedomain)',label="\$\\frac{d}{dt}\$",color=:green)
        end
        PyPlot.xlabel("\$\\omega\$")
        PyPlot.ylabel("a")
        PyPlot.legend(loc="lower left")

        
        PyPlot.subplot(313)
        PyPlot.grid()
        if EIS_IS
            PyPlot.plot(real(z_freqdomain),-imag(z_freqdomain),label="\$i\\omega\$", color=:red)
        end
        if EIS_TDS
            PyPlot.plot(real(z_timedomain),-imag(z_timedomain),label="\$\\frac{d}{dt}\$", color=:green)
        end
        PyPlot.xlabel("Re")
        PyPlot.ylabel("Im")
        PyPlot.legend(loc="lower center")
        PyPlot.tight_layout()
        pause(1.0e-10)
    end

    end
end


end
