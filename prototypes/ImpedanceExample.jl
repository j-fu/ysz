module ImpedanceExample

using LsqFit
using Printf
using VoronoiFVM

if installed("Plots")
    using Plots
end

#
# Calculate impedance in time domain
# 
function timedomain_impedance(sys, # time domain system
                              ω,   # frequency 
                              steadystate, # steady state slution
                              excited_bc,  # excitation bc number
                              excited_bcval, # excitation bc value
                              measurement_testfunc; # test function corresponding to measurement bc
                              excitation_amplitude=1.0e-4,  # amplitude of excitation
                              tref=0.0, # time step refinement level
                              tol_amplitude=1.0e-3, # tolerance for detection of settled amplitude
                              fit=false, # perform additional fit of amplitude+phase shift 
                              fit_window_size=20.0, # window size for additional fit
                              amplitude_plot=false, # Plot amplitude evolution
                              )
    tfac=0.1*2.0^(-tref)
    tstep=tfac/ω

    fit_window=0.0

    # obtaine measurement of steady state
    measured_steady=integrate(sys,measurement_testfunc,steadystate)[1]

    # solution arrays for time steping
    Uold=copy(steadystate)
    U=copy(steadystate)

    settled_amplitude=0.0 # value of settled amplitude (to be detected)
    t_settle=1.0e10  # time for setttling amplitude (to be detected)
    phase_shift=0.0    # phase shift  (to be detected)

    measured_min=0.0 # minimum of measurement 
    measured_max=0.0 # maximum of measurement
    t_current=0.0    # running time
    t_abort=10000.0/ω # emergency abort time
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
        sys.boundary_values[1,excited_bc]=excited_bcval+excitation_val
        solve!(U,Uold,sys,tstep=tstep)

        # Obtain measurement
        measured_val=(integrate(sys,measurement_testfunc,U, Uold,tstep)[1]-measured_steady)/excitation_amplitude
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
    if amplitude_plot
        p=plot(all_times,all_measured, label="measured",size=(600,800),legend=:bottomright,ylim=(-2*settled_amplitude,2*settled_amplitude))
        plot!(p,all_times,model(all_times,params), label="estimated")
        plot!(p,[t_settle, t_settle],[-settled_amplitude,0],linewidth=3,label="t_settle")
        gui(p)
    end
    z=settled_amplitude*exp(1im*phase_shift)
    @show z
    return z
end





function main(;nref=0, # spatial refinement
              tref=0,  # time domain refinement
              doplot=false, # result plot
              time_domain=false,  # compare to  time domain solution
              amplitude_plot=false, # plot amplitude during time domain solution
              fit_amplitude=false, # fit amplitude during time domain solution
              min_amplitude=1.0e-3, # minmal amplitude value (relative to excitation)
              R=1.0,      # Model datum: reaction coefficient
              Rexp=1.0,   # Model datum: reaction exponent
              D=1.0,      # Model datum: diffusion coefficient  
              Dexp=1.0    # Model datum: diffusion exponent     
              )
    # Domain size
    L=1.0
    # Storage coefficient
    C=1.0

    # Range of frequencies
    ω0=1.0
    ω1=1.0e4
    ωfac=1.3

    # Create array which is refined close to 0
    h0=0.1/2.0^nref
    h1=0.25/2.0^nref
    X=VoronoiFVM.geomspace(0.0,L,h0,h1)

    # Excited + measured bc
    excited_bc=1
    excited_bcval=1.0
    measured_bc=2

    
    # Create discretitzation grid
    grid=VoronoiFVM.Grid(X)


    # Linearity flag
    is_linear=true
    
    if Rexp!=1.0 || Dexp !=1.0
        is_linear=false
    end
    
    # Declare constitutive functions
    flux=function(f,u,edge,data)
        uk=viewK(edge,u)  
        ul=viewL(edge,u)
        f[1]=D*(uk[1]^Dexp-ul[1]^Dexp)
    end

    storage=function(f,u,node,data)
        f[1]=C*u[1]
    end

    reaction=function(f,u,node,data)
        f[1]=R*u[1]^Rexp
    end

    # Create physics struct
    physics=VoronoiFVM.Physics(flux=flux,
                               storage=storage,
                               reaction=reaction
                               )
    # Create discrete system and enabe species
    sys=VoronoiFVM.DenseSystem(grid,physics)
    enable_species!(sys,1,[1])

    # Create test function for current measurement
    factory=VoronoiFVM.TestFunctionFactory(sys)
    measurement_testfunction=testfunction(factory,[excited_bc],[measured_bc])

    # Set boundary values
    sys.boundary_values[1,excited_bc]=excited_bcval
    sys.boundary_values[1,measured_bc]=0.0
    
    sys.boundary_factors[1,excited_bc]=VoronoiFVM.Dirichlet
    sys.boundary_factors[1,measured_bc]=VoronoiFVM.Dirichlet

    # Solve for steady state
    inival=unknowns(sys)
    steadystate=unknowns(sys)
    inival.=0.0
    control=VoronoiFVM.NewtonControl()
    control.verbose=true
    control.max_iterations=400
    solve!(steadystate,inival,sys,control=control)
    control.verbose=false
    control.max_iterations=10

    isys=VoronoiFVM.ImpedanceSystem(sys,steadystate,1, excited_bc)

    # Prepare recording of impedance results
    all_omega=zeros(0)
    all_freqdomain=zeros(Complex{Float64},0)
    all_exact=zeros(Complex{Float64},0)
    all_timedomain=zeros(Complex{Float64},0)

    # Frequency loop
    ω=ω0
    UZ=unknowns(isys)
    while ω<ω1
        @show ω

        # solve impedance system
        solve!(UZ,isys,ω)

        # obtain measurement in frequency  domain
        z_freqdomain=integrate(isys,measurement_testfunction,ω,UZ)[1]
        
        # record approximate solution in frequency domain
        push!(all_omega, ω)
        push!(all_freqdomain,z_freqdomain)

        # Calculate exact solution (valid only in linear case!)
        iω=1im*ω
        z=sqrt(iω*C/D+R/D);
        eplus=exp(z*L);
        eminus=exp(-z*L);
        z_exact=2.0*D*z/(eminus-eplus);
        push!(all_exact,z_exact)

        # Perform time domain simulation
        if time_domain
            z_timedomain=timedomain_impedance(sys,ω,steadystate,excited_bc,excited_bcval,measurement_testfunction,
                                              amplitude_plot=amplitude_plot,
                                              tref=tref,
                                              fit=fit_amplitude)
            push!(all_timedomain,z_timedomain)
        end

        # finish if damping is below min_amplitude
        if abs(z_freqdomain)<min_amplitude
            break
        end

        ω=ω*ωfac
    end

    # plot result
    if doplot
        function positive_angle(z)
            ϕ=angle(z)
            if ϕ<0.0
                ϕ=ϕ+2*π
            end
            return ϕ
        end

        # Bode plot: phase
        p1=plot(grid=true,title="x=0",xlabel="omega", ylabel="phi", legend=:topleft,xaxis=:log)
        plot!(p1,all_omega,positive_angle.(all_freqdomain),label="freq. domain",marker=:circle,color=:green)
        if is_linear
            plot!(p1,all_omega,positive_angle.(all_exact),label="exact",marker=:cross,color=:red)
        end
        if time_domain
            plot!(p1,all_omega,positive_angle.(all_timedomain),label="time domain",color=:blue,marker=:square,markersize=2)
        end

        # Bode plot: absolute value
        p2=plot(grid=true,title="x=0",xlabel="ω", ylabel="A", legend=:bottomleft,xaxis=:log,yaxis=:log)
        plot!(p2,all_omega,abs.(all_freqdomain),label="freq. domain",marker=:circle,color=:green)
        if is_linear
            plot!(p2,all_omega,abs.(all_exact),label="exact",marker=:cross,color=:red)
        end
        if time_domain
            plot!(p2,all_omega,abs.(all_timedomain),label="time domain",color=:blue,marker=:square,markersize=2)
        end
        

        # Nyquist plot
        p3=plot(grid=true,title="x=L",xlabel="Re", ylabel="Im", legend=:bottomleft)
        plot!(p3,real(all_freqdomain),imag(all_freqdomain),label="freq. domain", marker=:circle,color=:green)
        if is_linear
            plot!(p3,real(all_exact),imag(all_exact),label="exact", marker=:cross,color=:red)
        end
        if time_domain
            plot!(p3,real(all_timedomain),imag(all_timedomain),label="time domain",color=:blue,marker=:square)
        end
        
        p=plot(p1,p2,p3,layout=(3,1),size=(600,800))
        gui(p)
    end
end


end

