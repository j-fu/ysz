module ImpedanceExample2

using Printf
using VoronoiFVM

if installed("Plots")
    using Plots
end


include("timedomain_impedance.jl")


function main(;nref=0, # spatial refinement
              tref=0,  # time domain refinement
              use_freq=true,
              doplot=false, # result plot
              time_domain=false,  # compare to  time domain solution
              plot_amplitude=false, # plot amplitude during time domain solution
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
    excited_bcval=1
    excited_spec=1
    measured_bc=2


    function meas_stdy(meas,u)
        sum=0.0
        for i=1:length(X)-1
            h=X[i+1]-X[i]
            ∇u= (u[i+1]-u[i])/h
            sum+=∇u^2*h
        end
        meas[1]=sum
    end
    
    function meas_tran(meas,u)
        meas[1]=0.0
    end
    
    
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


    dmeas_stdy=measurement_derivative(sys,meas_stdy,steadystate)
    dmeas_tran=measurement_derivative(sys,meas_tran,steadystate)
    
    isys=VoronoiFVM.ImpedanceSystem(sys,steadystate,excited_spec, excited_bc)

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
        # obtain measurement in frequency  domain
        z_freqdomain=freqdomain_impedance(isys,ω,steadystate,excited_spec,excited_bc,excited_bcval,dmeas_stdy, dmeas_tran)

        # record approximate solution in frequency domain
        push!(all_omega, ω)
        push!(all_freqdomain,z_freqdomain)

        # Calculate exact solution (valid only in linear case!)
        iω=1im*ω
        z=sqrt(iω*C/D+R/D);
        d=sqrt(R/D)

        eplus=exp(z*L);
        eminus=exp(-z*L);

        dplus=exp(d*L);
        dminus=exp(-d*L);

        a=eminus/(eminus-eplus)
        b=eplus/(eplus-eminus)
        f=dminus/(dminus-dplus)
        g=dplus/(dplus-dminus)

        AA(x)=exp((z+d)*x)*a*f/(z+d)
        BB(x)=exp((z-d)*x)*a*g/(z-d)
        CC(x)=exp((d-z)*x)*b*f/(d-z)
        DD(x)=exp(-(z+d)*x)*b*g/(z+d)

        INT(x)=z*d*(AA(x)-BB(x)-CC(x)-DD(x))

        z_exact=2.0(INT(L)- INT(0))

        @show z_exact, z_freqdomain
        
        push!(all_exact,z_exact)

        # Perform time domain simulation
        if time_domain
            z_timedomain=timedomain_impedance(sys,ω,steadystate,excited_spec,excited_bc,excited_bcval,meas_stdy,meas_tran,
                                              plot_amplitude=plot_amplitude,
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
        if use_freq
            plot!(p1,all_omega,positive_angle.(all_freqdomain),label="freq. domain",marker=:circle,color=:green)
        end
        if is_linear
            plot!(p1,all_omega,positive_angle.(all_exact),label="exact",marker=:cross,color=:red)
        end
        if time_domain
            plot!(p1,all_omega,positive_angle.(all_timedomain),label="time domain",color=:blue,marker=:square,markersize=2)
        end

        # Bode plot: absolute value
        p2=plot(grid=true,title="x=0",xlabel="ω", ylabel="A", legend=:bottomleft,xaxis=:log,yaxis=:log)
        if use_freq
            plot!(p2,all_omega,abs.(all_freqdomain),label="freq. domain",marker=:circle,color=:green)
        end
        if is_linear
            plot!(p2,all_omega,abs.(all_exact),label="exact",marker=:cross,color=:red)
        end
        if time_domain
            plot!(p2,all_omega,abs.(all_timedomain),label="time domain",color=:blue,marker=:square,markersize=2)
        end
        

        # Nyquist plot
        p3=plot(grid=true,title="x=L",xlabel="Re", ylabel="Im", legend=:bottomleft)
        if use_freq
            plot!(p3,real(all_freqdomain),imag(all_freqdomain),label="freq. domain", marker=:circle,color=:green)
        end
        if is_linear
            plot!(p3,real(all_exact),imag(all_exact),label="exact", marker=:cross,color=:red)
        end
        if time_domain
            plot!(p3,real(all_timedomain),imag(all_timedomain),label="time domain",color=:blue,marker=:square)
        end
        
        p=plot(p1,p2,p3,layout=(3,1),size=(600,800))
        gui(p)
    end
   return all_freqdomain[end]
end
function test()
    main()≈2.533912747815139 + 0.0017950992333295279im
end

end

