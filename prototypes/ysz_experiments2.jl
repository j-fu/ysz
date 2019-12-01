module ysz_experiments2

using Printf
using VoronoiFVM
using PyPlot
using DataFrames
using CSV
using LeastSquaresOptim

##########################################
# internal import of YSZ repo ############

include("../src/ysz_model_fitted_parms.jl")
include("../prototypes/timedomain_impedance.jl")


iphi=ysz_model_fitted_parms.iphi
iy=ysz_model_fitted_parms.iy
ib=ysz_model_fitted_parms.ib


#using ysz_model_fitted_parms
#const label_ysz_model = ysz_model_fitted_parms

# --------- end of YSZ import ---------- #
##########################################


function run_new(;test=false, print_bool=false, debug_print_bool=false,
                 verbose=false ,pyplot=false, save_files=false, width=10.0e-9,
                 dx_exp=-9, tref=0, voltammetry=false, EIS_TDS=false, EIS_IS=false,
                 EIS_make_plots=false , dlcap=false, voltrate=0.005,
                 upp_bound=0.5, low_bound=-0.5, sample=40,
                 prms_in=[21.71975544711280, 20.606423236896422, 0.0905748, -0.708014, 0.6074566741435283, 0.1], nu_in=0.9)

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
    X=width*VoronoiFVM.geomspace(0.0,1.0,dx_start,1e-1)
    #println("X = ",X)
    #
    grid=VoronoiFVM.Grid(X)
    #
    
    parameters=ysz_model_fitted_parms.YSZParameters()
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
    parameters = ysz_model_fitted_parms.YSZParameters_update(parameters)

    physics=VoronoiFVM.Physics(
        data=parameters,
        num_species=3,
        storage=ysz_model_fitted_parms.storage!,
        flux=ysz_model_fitted_parms.flux!,
        reaction=ysz_model_fitted_parms.reaction!,
        breaction=ysz_model_fitted_parms.breaction!,
        bstorage=ysz_model_fitted_parms.bstorage!
    )
    #
    if print_bool
        ysz_model_fitted_parms.printfields(parameters)
    end

    sys=VoronoiFVM.SparseSystem(grid,physics)
    #sys=VoronoiFVM.DenseSystem(grid,physics)
    enable_species!(sys,iphi,[1])
    enable_species!(sys,iy,[1])
    enable_boundary_species!(sys,ib,[1])


    #
    #sys.boundary_values[iphi,1]=1.0e-0
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
    
    phi0 = ysz_model_fitted_parms.equil_phi(parameters)
    if print_bool
        println("phi0 = ",phi0)
    end
    if dlcap
        phi0 = 0
        if print_bool
            println("dlcap > phi0= ", phi0)
        end
    end


    for inode=1:size(inival,2)
        inival[iphi,inode]=0.0
        inival[iy,inode]= parameters.y0
    end
    inival[ib,1]=parameters.y0

    #
    control=VoronoiFVM.NewtonControl()
    control.verbose=verbose
    control.tol_linear=1.0e-4
    control.tol_relative=1.0e-5
    #control.tol_absolute=1.0e-4
    #control.max_iterations=3
    control.max_lureuse=0
    control.damp_initial=1.0e-5
    control.damp_growth=1.9
    time=0.0
 
    
    
    #
    # Transient part of measurement functional 
    #
    function meas_tran(meas, u)
        U=reshape(u,sys)
        Qb= - integrate(sys,ysz_model_fitted_parms.reaction!,U) # \int n^F            
        dphi_end = U[iphi, end] - U[iphi, end-1]
        dx_end = X[end] - X[end-1]
        dphiB=parameters.eps0*(1+parameters.chi)*(dphi_end/dx_end)
        Qs= (parameters.e0/parameters.areaL)*parameters.zA*U[ib,1]*parameters.ms_par*(1-parameters.nus) # (e0*zA*nA_s)
        meas[1]=-Qs[1]-Qb[iphi]
    end

    #
    # Steady part of measurement functional
    #
    function meas_stdy(meas, u)
        U=reshape(u,sys)
        meas[1]=-2*parameters.e0*ysz_model_fitted_parms.electroreaction(parameters, U[ib,1])
    end

    #
    # The overall measurement (in the time domain) is meas_stdy(u)+ d/dt meas_tran(u)
    #

    # Calculate steady state solution
    steadystate = unknowns(sys)

    
    phi_OCV = 0.0374762
    
    excited_spec=iphi
    excited_bc=1
    excited_bcval=phi_OCV
    
    # relaxation
    sys.boundary_values[iphi,1]= phi_OCV
    solve!(steadystate,inival,sys, control=control)


    # Create impedance system
    isys=VoronoiFVM.ImpedanceSystem(sys,steadystate,excited_spec, excited_bc)

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
            zfreq=freqdomain_impedance(isys,w,steadystate,excited_spec,excited_bc,excited_bcval,dmeas_stdy, dmeas_tran)
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
            plot(real(z_freqdomain),-imag(z_freqdomain),label="\$i\\omega\$", color=:red)
        end
        if EIS_TDS
            plot(real(z_timedomain),-imag(z_timedomain),label="\$\\frac{d}{dt}\$", color=:green)
        end
        PyPlot.xlabel("Re")
        PyPlot.ylabel("Im")
        PyPlot.legend(loc="lower center")
        PyPlot.tight_layout()
        pause(1.0e-10)
    end


    #########################################
    #########################################
    #########################################
    #########################################
    #########################################
    #########################################
    #########################################
    #########################################
    
    # code for performing CV
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

        U = unknowns(sys)
        U0 = unknowns(sys)
        if test
            U .= inival
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
            solve!(U,inival,sys, control=control,tstep=tstep)
            Qb= - integrate(sys,ysz_model_fitted_parms.reaction!,U) # \int n^F            
            dphi_end = U[iphi, end] - U[iphi, end-1]
            dx_end = X[end] - X[end-1]
            dphiB=parameters.eps0*(1+parameters.chi)*(dphi_end/dx_end)
            Qs= (parameters.e0/parameters.areaL)*parameters.zA*U[ib,1]*parameters.ms_par*(1-parameters.nus) # (e0*zA*nA_s)

                    
            # for faster computation, solving of "dtstep problem" is not performed
            U0 .= inival
            inival.=U
            Qb0 = - integrate(sys,ysz_model_fitted_parms.reaction!,U0) # \int n^F
            dphi0_end = U0[iphi, end] - U0[iphi, end-1]
            dphiB0 = parameters.eps0*(1+parameters.chi)*(dphi0_end/dx_end)
            Qs0 = (parameters.e0/parameters.areaL)*parameters.zA*U0[ib,1]*parameters.ms_par*(1-parameters.nus) # (e0*zA*nA_s)


            
            # time derivatives
            Is  = - (Qs[1] - Qs0[1])/tstep                
            Ib  = - (Qb[iphi] - Qb0[iphi])/tstep 
            Ibb = - (dphiB - dphiB0)/tstep
            
            
            # reaction average
            reac = - 2*parameters.e0*ysz_model_fitted_parms.electroreaction(parameters, U[ib,1])
            reacd = - 2*parameters.e0*ysz_model_fitted_parms.electroreaction(parameters,U0[ib,1])
            Ir= 0.5*(reac + reacd)

            #############################################################
            #multiplication by area of electrode I = A * ( ... )
            Ibb = Ibb*AreaEllyt
            Ib = Ib*AreaEllyt
            Is = Is*AreaEllyt
            Ir = Ir*AreaEllyt
            #
            
            if debug_print_bool
                @printf("t = %g     U = %g   state = %s  reac = %g  \n", istep*tstep, phi, state, Ir)
            end
            
            # storing data
            append!(y0_range,U[iy,1])
            append!(ys_range,U[ib,1])
            append!(phi_range,phi)
            #
            append!(Ib_range,Ib)
            append!(Is_range,Is)
            append!(Ibb_range,Ibb)
            append!(Ir_range, Ir)
            #
            append!(time_range,tstep*istep)
            
            if state=="cv_is_on"
                if save_files
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
                    plot((10^9)*X[:],U[iphi,:],label="phi (V)")
                    plot((10^9)*X[:],U[iy,:],label="y")
                    plot(0,U[ib,1],"go", markersize=ys_marker_size, label="y_s")
                    l_plot = 5.0
                    PyPlot.xlim(-0.01*l_plot, l_plot)
                    PyPlot.ylim(-0.5,1.1)
                    PyPlot.xlabel("x (nm)")
                    PyPlot.legend(loc="best")
                    PyPlot.grid()
                end
                
                #if num_subplots > 1
                #    subplot(num_subplots*100 + 12)
                #    plot((10^3)*X[:],U[iphi,:],label="phi (V)")
                #    plot((10^3)*X[:],U[iy,:],label="y")
                #    plot(0,U[ib,1],"go", markersize=ys_marker_size, #label="y_s")
                #    PyPlot.ylim(-0.5,1.1)
                #    PyPlot.xlabel("x (mm)")
                #    PyPlot.legend(loc="best")
                #    PyPlot.grid()
                #end
                
                if (num_subplots > 1) && (istep_cv_start > -1)
                    cv_range = (istep_cv_start+1):length(phi_range)
                    subplot(num_subplots*100 + 12)
                    plot(phi_range[cv_range].-phi0, ((Is_range + Ib_range + Ir_range + Ibb_range)[cv_range]) ,label="total current")
                    
                    PyPlot.xlabel(L"\eta \ (V)")
                    PyPlot.ylabel(L"I \ (A)")
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

            out_data_dir = "./results/CV_data/"
            
            if !ispath(out_data_dir)
                mkpath(out_data_dir)
            end

            CSV.write(string(out_data_dir, out_name,".csv"),out_df)
            
            
            if pyplot
                out_fig_dir = "./results/CV_images/"
            
                if !ispath(out_fig_dir)
                    mkpath(out_fig_dir)
                end
            
                PyPlot.savefig(string(out_fig_dir, out_name,".png"))
            end
        end
        if test
            I1 = integrate(sys, ysz_model_fitted_parms.reaction!, U)
            #println(I1)
            return I1[1]
        end
    end
end


end
