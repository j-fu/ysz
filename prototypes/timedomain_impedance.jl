using LsqFit

#
# Calculate impedance in time domain
# 
function timedomain_impedance(sys, # time domain system
                              ω,   # frequency 
                              steadystate, # steady state slution
                              excited_spec,  # excitated spec
                              excited_bc,  # excitation bc number
                              excited_bcval, # excitation bc value
                              meas_stdy,meas_tran;
                              excitation_amplitude=1.0e-6,  # amplitude of excitation
                              tref=0.0, # time step refinement level
                              tol_amplitude=1.0e-3, # tolerance for detection of settled amplitude
                              fit=false, # perform additional fit of amplitude+phase shift 
                              fit_window_size=20.0, # window size for additional fit
                              plot_amplitude=false, # Plot amplitude evolution
                              )
    tfac=0.1*2.0^(-tref)
    tstep=tfac/ω

    fit_window=0.0

    if !installed("Plots")
        plot_amplitude=false
    end
    
    # obtaine measurement of steady state
    # mstdy_steadystate=integrate(sys,measurement_testfunc,steadystate)[1]
    mstdy_steadystate=[0.0]
    meas_stdy(mstdy_steadystate,values(steadystate))
    
    # solution arrays for time steping
    Uold=copy(steadystate)
    U=copy(steadystate)

    settled_amplitude=0.0 # value of settled amplitude (to be detected)
    t_settle=1.0e10  # time for setttling amplitude (to be detected)
    phase_shift=0.0    # phase shift  (to be detected)

    measured_min=0.0 # minimum of measurement 
    measured_max=0.0 # maximum of measurement
    t_current=0.0    # running time
    t_abort=5000.0/ω # emergency abort time
    t_phase_start=0.0 # time of start of phase
    excitation_prev=0.0 # old excitation value
    measured_prev=0.0  # old meassurement value

    # Data arrays (for plotting)
    all_times=[]
    all_measured=[]
    fit_times=[]
    fit_measured=[]

    # set fit window
    if fit
        fit_window=fit_window_size/ω
    end

    # time loop: we loop until amplitude has settled. Optionally
    # append the fit window
    while t_current<t_settle+fit_window 
        # new time
        t_current=t_current+tstep 

        # solve with  new excitation value
        excitation_val=sin(ω*t_current)*excitation_amplitude
        sys.boundary_values[excited_spec,excited_bc]=excited_bcval+excitation_val
        solve!(U,Uold,sys,tstep=tstep)

        # Obtain measurement
        # measured_val=(integrate(sys,measurement_testfunc,U, Uold,tstep)[1]-mstdy_steadystate)/excitation_amplitude
        mstdy_U=[0.0]
        meas_stdy(mstdy_U,values(U))
        mtran_U=[0.0]
        meas_tran(mtran_U,values(U))
        mtran_Uold=[0.0]
        meas_tran(mtran_Uold,values(Uold))
        
        measured_val=(mstdy_U[1] + (mtran_U[1]-mtran_Uold[1])/tstep-mstdy_steadystate[1])/excitation_amplitude
        
        
        
        push!(all_times,t_current)
        push!(all_measured,measured_val)

        # Detect begin of new phase: excitation passes zero in negative direction
        if excitation_val*excitation_prev<=0.0 && excitation_val<=excitation_prev
            # Set phase start
            t_phase_start=t_current

            # Set measurement minmax value for phase begin
            measured_min=measured_val
            measured_max=measured_val
        end

        # update measureing minmax for current phase
        measured_min=min(measured_min,measured_val)
        measured_max=max(measured_max,measured_val)
        
        # Detect phase shift: measurement passes zero in negative direction
        if measured_val*measured_prev<=0.0 &&  measured_val<=measured_prev
            # phase shift obtained from difference to start of phase
            phase_shift=-(t_current-t_phase_start)*ω
        end

        # figure out if amplitude has settled:
        # min and max of measurement have different sign,but nearly the same absolute value.
        if t_settle>1.0e9 &&  measured_max* measured_min <0.0 && t_current*ω>4π
            relative_amplitude_mismatch=(abs(measured_max)-abs(measured_min))/(abs(measured_min)+abs(measured_max))
            if abs(relative_amplitude_mismatch)<tol_amplitude
                t_settle=t_current
                settled_amplitude=measured_max
            end
        end

        # if we continue for fitting after t_settle, store the correspondig data
        if t_current>t_settle
            push!(fit_times,t_current)
            push!(fit_measured,measured_val)
        end

        # prepare next time step
        Uold.=U
        excitation_prev=excitation_val
        measured_prev=measured_val
        # emergency abort
        if t_current>t_abort
            error("reached t_abort without detecting amplitude")
        end
    end

    # curve_fit to data obtained since t_settle
    # it seems we do not need this.
    # Severe problems without proper initial guess for phase...
    @. model(t,p)=p[1]*sin(ω*t+p[2])
    params=[settled_amplitude,phase_shift,0.0]
    if fit
        fit=curve_fit(model,fit_times,fit_measured,params)
        params=coef(fit)
        settled_amplitude=params[1]
        phase_shift=params[2]
    end

    # Optional plot of amplitude
    if plot_amplitude
        p=plot(all_times,all_measured, label="measured",size=(600,800),legend=:bottomright,ylim=(-2*settled_amplitude,2*settled_amplitude))
        plot!(p,all_times,model(all_times,params), label="estimated")
        plot!(p,[t_settle, t_settle],[-settled_amplitude,0],linewidth=3,label="t_settle")
        gui(p)
    end
    z=settled_amplitude*exp(1im*phase_shift)
    return z
end


