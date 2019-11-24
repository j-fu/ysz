module ysz_experiments

using Printf
using VoronoiFVM
using PyPlot
using DataFrames
using CSV
using LeastSquaresOptim

##########################################
# internal import of YSZ repo ############
cur_dir=pwd()
cd("../src")
src_dir=pwd()
cd(cur_dir)

push!(LOAD_PATH,src_dir)

using ysz_model_fitted_parms
const label_ysz_model = ysz_model_fitted_parms

# --------- end of YSZ import ---------- #
##########################################


function run_new(;test=false, print_bool=false, debug_print_bool=false, verbose=false ,pyplot=false, save_files=false, width=10.0e-9,  dx_exp=-9, voltammetry=false, EIS_TDS=false, EIS_IS=false, EIS_make_plots=false , dlcap=false, voltrate=0.005, upp_bound=0.5, low_bound=-0.5, sample=40, prms_in=[21.71975544711280, 20.606423236896422, 0.0905748, -0.708014, 0.6074566741435283, 0.1], nu_in=0.9)

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
    
    parameters=label_ysz_model.YSZParameters()
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
    parameters = label_ysz_model.YSZParameters_update(parameters)

    physics=VoronoiFVM.Physics(
        data=parameters,
        num_species=3,
        storage=label_ysz_model.storage!,
        flux=label_ysz_model.flux!,
        reaction=label_ysz_model.reaction!,
        breaction=label_ysz_model.breaction!,
        bstorage=label_ysz_model.bstorage!
    )
    #
    if print_bool
        label_ysz_model.printfields(parameters)
    end

    #sys=VoronoiFVM.SparseSystem(grid,physics)
    sys=VoronoiFVM.DenseSystem(grid,physics)
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
    
    phi0 = label_ysz_model.equil_phi(parameters)
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
 
 
    ######## code for performing EIS ########
    #########################################
    #########################################
    #########################################
    #########################################
    #########################################
    
        
    # EIS - VoronoiFVM.ImpedanceSystem
    if EIS_IS
        U = unknowns(sys)
        
        phi_OCV = 0.0374762
        
        
        # relaxation
        sys.boundary_values[iphi,1]= phi_OCV
        solve!(U,inival,sys, control=control)

        
        factory=VoronoiFVM.TestFunctionFactory(sys)
        #tf0=testfunction(factory,[2],[1])
        tfL=testfunction(factory,[1],[2])
            
        excited_spec=iphi
        excited_bc=1
        isys=VoronoiFVM.ImpedanceSystem(sys,U,excited_spec, excited_bc)
        
        w0 = 1.0e-6
        w1 = 1.0e6
        
        w = w0
        
        UZ=unknowns(isys)
        
        allIL=zeros(Complex{Float64},0)
        
        while w<w1
            solve!(UZ,isys,w)
            
            IL=integrate(isys,tfL,w,UZ)[1]
            
            push!(allIL,IL)
            
            # growth factor such that there are 10 points in every order of magnitude
            # (which is consistent with "freq" list below)
            w=w*1.25892
        end
        
        PyPlot.clf()
        PyPlot.grid()
        plot(real(allIL),imag(allIL),label="calc")
        PyPlot.legend(loc="upper left")
        pause(1.0e-10)
        #waitforbuttonpress()
    end



    #########################################
    #########################################
    #########################################
    #########################################
    #########################################
    
    # EIS - time-dependent simulation
    if EIS_TDS
        if print_bool
            println("performing EIS time-dependent simulation ...")  
        end
        if EIS_make_plots
            save_files = true
        end
                
        freq = [1.0E-06 1.3E-06 1.6E-06 2.0E-06 2.5E-06 3.2E-06 4.0E-06 5.0E-06 6.3E-06 7.9E-06 1.0E-05 1.3E-05 1.6E-05 2.0E-05 2.5E-05 3.2E-05 4.0E-05 5.0E-05 6.3E-05 7.9E-05 1.0E-04 1.3E-04 1.6E-04 2.0E-04 2.5E-04 3.2E-04 4.0E-04 5.0E-04 6.3E-04 7.9E-04 1.0E-03 1.3E-03 1.6E-03 2.0E-03 2.5E-03 3.2E-03 4.0E-03 5.0E-03 6.3E-03 7.9E-03 1.0E-02 1.3E-02 1.6E-02 2.0E-02 2.5E-02 3.2E-02 4.0E-02 5.0E-02 6.3E-02 7.9E-02 1.0E-01 1.3E-01 1.6E-01 2.0E-01 2.5E-01 3.2E-01 4.0E-01 5.0E-01 6.3E-01 7.9E-01 1.0E+00 1.3E+00 1.6E+00 2.0E+00 2.5E+00 3.2E+00 4.0E+00 5.0E+00 6.3E+00 7.9E+00 1.0E+01 1.3E+01 1.6E+01 2.0E+01 2.5E+01 3.2E+01 4.0E+01 5.0E+01 6.3E+01 7.9E+01 1.0E+02 1.3E+02 1.6E+02 2.0E+02 2.5E+02 3.2E+02 4.0E+02 5.0E+02 6.3E+02 7.9E+02 1.0E+03 1.3E+03 1.6E+03 2.0E+03 2.5E+03 3.2E+03 4.0E+03 5.0E+03 6.3E+03 7.9E+03 1.0E+04 1.3E+04 1.6E+04 2.0E+04 2.5E+04 3.2E+04 4.0E+04 5.0E+04 6.3E+04 7.9E+04 1.0E+05 1.3E+05 1.6E+05 2.0E+05 2.5E+05 3.2E+05 4.0E+05 5.0E+05 6.3E+05 7.9E+05 1.0E+06]        
        
        #freq = freq[1:5]
        
        file_label = string("EIS-test")
        out_dir = string("./results/EIS_aux_data/",file_label)
        
        for f in freq
            ##############################################################################
            ## Control panel #############################################################
            
            pp = 3 			# number of periods
            pbp = 10      		# number of "time" points per 1 periode
            t_eis_start = 1 /f	# nominator says after how many cycles the recording of EIS data starts
            
            phi_OCV = 0.0374762	# [V]	
            
            eis_amplitude = 0.005 	# [V]
            
            ##############################################################################
            ##############################################################################
            if print_bool
                println("frequency ",f)
            end
            out_name=string(file_label, "_f", @sprintf("%.1e",f))
        
            istep=0


            # inicializing storage lists
            y0_range=zeros(0)
            ys_range=zeros(0)
            phi_range=zeros(0)
            #
            Is_range=zeros(0)
            Ib_range=zeros(0)
            Ibb_range=zeros(0)
            Ir_range=zeros(0)
            
            time_range = zeros(0)  # [s]
            
            if save_files
                out_df = DataFrame(U = Float64[], Itot = Float64[])
            end
            
            U = unknowns(sys)
            U0 = unknowns(sys)
            
            w = 2*pi*f
            T_start = 0.0
            tstep = 1/(f*pbp)
            T_end = pp/f
            
            t = T_start
                    

            allt = zeros(0)
            allphi = zeros(0)
            allI = zeros(0)
            
            istep_eis_start = -1
               
            # relaxation
            sys.boundary_values[iphi,1]=phi_OCV
            solve!(U,inival,sys, control=control)
        
        
            # eis
            while t <= T_end
                istep+= 1
                t = t + tstep
                #println(t)
                if (t > t_eis_start) && (istep_eis_start < 0)
                    istep_eis_start = istep
                end
                
                phi = eis_amplitude*sin(w*t) + phi_OCV
                
                push!(allt,t)
                push!(allphi,phi)
                #println(phi)
                #println(X)
                
                
                sys.boundary_values[iphi,1]=phi
                
                #println("bound ",sys.boundary_values[iphi,1]," inival ",inival)
                
                solve!(U,inival,sys, control=control,tstep=tstep)
                Qb= - integrate(sys,label_ysz_model.reaction!,U) # \int n^F            
                dphi_end = U[iphi, end] - U[iphi, end-1]
                dx_end = X[end] - X[end-1]
                dphiB=parameters.eps0*(1+parameters.chi)*(dphi_end/dx_end)
                Qs= (parameters.e0/parameters.areaL)*parameters.zA*U[ib,1]*parameters.ms_par*(1-parameters.nus) # (e0*zA*nA_s)

                        
                # for faster computation, solving of "dtstep problem" is not performed
                U0 .= inival
                inival.=U
                Qb0 = - integrate(sys,label_ysz_model.reaction!,U0) # \int n^F
                dphi0_end = U0[iphi, end] - U0[iphi, end-1]
                dphiB0 = parameters.eps0*(1+parameters.chi)*(dphi0_end/dx_end)
                Qs0 = (parameters.e0/parameters.areaL)*parameters.zA*U0[ib,1]*parameters.ms_par*(1-parameters.nus) # (e0*zA*nA_s)


                
                # time derivatives
                Is  = - (Qs[1] - Qs0[1])/tstep                
                Ib  = - (Qb[iphi] - Qb0[iphi])/tstep 
                Ibb = - (dphiB - dphiB0)/tstep
                
                
                # reaction average
                reac = - 2*parameters.e0*label_ysz_model.electroreaction(parameters, U[ib,1])
                reacd = - 2*parameters.e0*label_ysz_model.electroreaction(parameters,U0[ib,1])
                Ir= 0.5*(reac + reacd)

                #############################################################
                #multiplication by area of electrode I = A * ( ... )
                #Ibb = Ibb*AreaEllyt
                #Ib = Ib*AreaEllyt
                #Is = Is*AreaEllyt
                #Ir = Ir*AreaEllyt
                #
                
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
                
                if save_files && (istep_eis_start > -1 )
                    if dlcap
                        push!(out_df,[phi-phi_OCV(Ib+Is+Ir)/voltrate    ])
                    else
                        push!(out_df,[phi-phi_OCV    Ib+Is+Ir   ])
                    end
                end

                
                        
                if (istep_eis_start > -1) && (pyplot)
                    
                    clf()
                    subplot(211)
                    plot(time_range[istep_eis_start:end],allphi[istep_eis_start:end])
                    subplot(212)
                    plot(time_range[istep_eis_start:end],(Ib_range + Is_range + Ibb_range + Ir_range)[istep_eis_start:end])
                    pause(1.0e-10)
                    #println(" -- ")
                end

            end
            
            if save_files                
                #println(out_dir)
                if !ispath(out_dir)
                    mkpath(out_dir)
                end
                
                CSV.write(string(out_dir,"/",out_name,".csv"),out_df)
                #if pyplot
                #    PyPlot.savefig(string("./images/",out_name,".png"))
                #end
            end
        end
        if EIS_make_plots
            run(`python _1D-BODE.py $file_label`)
        end
    end

    
    
    
    
    
    
    
    
    
    #########################################
    #########################################
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
            Qb= - integrate(sys,label_ysz_model.reaction!,U) # \int n^F            
            dphi_end = U[iphi, end] - U[iphi, end-1]
            dx_end = X[end] - X[end-1]
            dphiB=parameters.eps0*(1+parameters.chi)*(dphi_end/dx_end)
            Qs= (parameters.e0/parameters.areaL)*parameters.zA*U[ib,1]*parameters.ms_par*(1-parameters.nus) # (e0*zA*nA_s)

                    
            # for faster computation, solving of "dtstep problem" is not performed
            U0 .= inival
            inival.=U
            Qb0 = - integrate(sys,label_ysz_model.reaction!,U0) # \int n^F
            dphi0_end = U0[iphi, end] - U0[iphi, end-1]
            dphiB0 = parameters.eps0*(1+parameters.chi)*(dphi0_end/dx_end)
            Qs0 = (parameters.e0/parameters.areaL)*parameters.zA*U0[ib,1]*parameters.ms_par*(1-parameters.nus) # (e0*zA*nA_s)


            
            # time derivatives
            Is  = - (Qs[1] - Qs0[1])/tstep                
            Ib  = - (Qb[iphi] - Qb0[iphi])/tstep 
            Ibb = - (dphiB - dphiB0)/tstep
            
            
            # reaction average
            reac = - 2*parameters.e0*label_ysz_model.electroreaction(parameters, U[ib,1])
            reacd = - 2*parameters.e0*label_ysz_model.electroreaction(parameters,U0[ib,1])
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
            I1 = integrate(sys, label_ysz_model.reaction!, U)
            #println(I1)
            return I1[1]
        end
    end
end


end
