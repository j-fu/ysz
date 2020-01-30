# # Testing adsorption fluxes
#
# Fork of Example # 160: Unipolar degenerate drift-diffusion
#
# ([source code](SOURCE_URL))
module Example160_UnipolarDriftDiffusion1D

using Printf

using VoronoiFVM
using Plots
using Profile
using PyPlot

mutable struct Data <: VoronoiFVM.AbstractData
    eps::Float64 
    z::Float64
    ic::Int32
    iphi::Int32
    
    DGR::Float64
    kR::Float64
    ibc::Int32

    names::Array{String,1}
    Data()=new()
end

function plot_solution(Plots,sys,U0, Ub,T)
    !isplots(Plots) && return
    ildata=data(sys[1])
    iphi=ildata.iphi
    ic=ildata.ic
    p1=Plots.plot(grid=true, title="bulk y, \\phi")
    p2=Plots.plot(grid=true, title="y_s(t)")
    p3=Plots.plot(grid=true, title="relative y_s(t)")
    name = ildata.names #["lin","exp","LoMA"]
    @views begin
      for (ii, zyz) in enumerate(sys)
          Plots.plot!(p1,zyz.grid.coord[1,:],U0[ii][iphi,:], label=string("Potential ",name[ii]))
          Plots.plot!(p1,zyz.grid.coord[1,:],U0[ii][ic,:], label=string("C- ",name[ii]), marker = 1)
          Plots.plot!(p1,[0],[Ub[ii][end]], label="",  marker=2)
          Plots.plot!(p2,T,Ub[ii], label=string("bC- ",name[ii]))
          Plots.plot!(p3,T,Ub[ii]./Ub[1], label=string("bC- ",name[ii]))
        end
    end
    p=Plots.plot(p1,p2,p3, layout=(3,1),legend=true, dpi=150)
    Plots.gui(p)
end




function classflux!(f,u,edge,data)
    uk=viewK(edge,u)
    ul=viewL(edge,u)
    ic=data.ic
    iphi=data.iphi
    f[iphi]=data.eps*(uk[iphi]-ul[iphi])
    arg=uk[iphi]-ul[iphi]
    bp,bm=fbernoulli_pm(uk[iphi]-ul[iphi])
    f[ic]=bm*uk[ic]-bp*ul[ic]
end 


function storage!(f,u,node,data)
    ic=data.ic
    iphi=data.iphi
    f[iphi]=0
    f[ic]=u[ic]
end

function bstorage!(f,u,node,data)
    ibc=data.ibc
    if node.region==1
        f[ibc]= u[ibc]
    end
end


function reaction!(f,u,node,data)
    ic=data.ic
    iphi=data.iphi
    f[iphi]=data.z*(1-2*u[ic])
    f[ic]=0
end

function breaction_testing!(f,u,node,data)
    ibc=data.ibc
    ic=data.ic
    iphi=data.iphi
    if node.region==1
        f[iphi]=0
        f[ic]=0 
        f[ibc]=0
    else
        f[iphi]=0
        f[ic]=0 
        f[ibc]=0
    end
end

#=
# #

```math
C + bV rightarrow bC + V\\
```

```math
r = k_R \left( c(1-c_b) - e^{-DGR}c_b(1-c) \right)
```

# Law of Mass Action adsorption
```math
r = k_R \left(e^{-0.5 \Delta G_R} c(1-c_b) - e^{0.5 \Delta G_R}c_b(1-c) \right)
```
=#
# function rate(u,data)
#     ibc=data.ibc
#     ic=data.ic
#     iphi=data.iphi
#     frate = data.kR*u[ic]*(1.0-u[ibc]) 
#     brate = data.kR*exp(-data.DGR)*u[ibc]*(1.0-u[ic]) 
#     return [frate,brate]
# end

function LoMA_adsorption!(f,u,node,data)
    ibc=data.ibc
    ic=data.ic
    iphi=data.iphi
    if node.region==1
        rate = data.kR*(exp(-0.5*data.DGR)*u[ic]*(1.0-u[ibc]) - exp(0.5*data.DGR)*u[ibc]*(1.0-u[ic]) )
        f[iphi]=0
        f[ic]= rate
        f[ibc]= -rate
    else
        f[iphi]=0
        f[ic]=0 
        f[ibc]=0
    end
end
#=
# exp_adsorption
```math
r = k_R \left( 
e^{-0.5 \Delta G_R}\left[\frac{c(1-c_b)}c_b(1-c)\right]^0.5 
-e^{-0.5 \Delta G_R}\left[\frac{c(1-c_b)}c_b(1-c)\right]^-0.5 
\right)
```
=#

function exp_adsorption!(f,u,node,data)
    ibc=data.ibc
    ic=data.ic
    iphi=data.iphi
    if node.region==1
        rate = data.kR*(
                        exp(-0.5*data.DGR)*(u[ic]*(1.0-u[ibc]))^0.5*(u[ibc]*(1.0-u[ic]))^-0.5
                        - exp(0.5*data.DGR)*(u[ibc]*(1.0-u[ic]))^0.5*(u[ic]*(1.0-u[ibc]))^-0.5
                )
        f[iphi]=0
        f[ic]= rate
        f[ibc]= -rate
    else
        f[iphi]=0
        f[ic]=0 
        f[ibc]=0
    end
end
#=
# Linear adsorption
```math
r = k_R \left( - e^{-DGR}+ log\left[\frac{c(1-c_b)}{c_b(1-c)}\right] \right)
```
=#

function linear_adsorption!(f,u,node,data)
    ibc=data.ibc
    ic=data.ic
    iphi=data.iphi
    if node.region==1
        rate = data.kR*(-data.DGR + log(u[ic]*(1.0-u[ibc])) - log(u[ibc]*(1.0-u[ic])) )
        f[iphi]=0
        f[ic]= rate
        f[ibc]= -rate
    else
        f[iphi]=0
        f[ic]=0 
        f[ibc]=0
    end
end

function sedanflux!(f,u,edge,data)
    uk=viewK(edge,u)
    ul=viewL(edge,u)
    ic=data.ic
    iphi=data.iphi
    f[iphi]=data.eps*(uk[iphi]-ul[iphi])
    muk=-log(1-uk[ic])
    mul=-log(1-ul[ic])
    bp,bm=fbernoulli_pm(data.z*2*(uk[iphi]-ul[iphi])+(muk-mul))
    f[ic]=bm*uk[ic]-bp*ul[ic]
end 

function quadraticDif_sedanflux!(f,u,edge,data)
    uk=viewK(edge,u)
    ul=viewL(edge,u)
    # TBD
end 

function vacancy_sedanflux!(f,u,edge,data)
    uk=viewK(edge,u)
    ul=viewL(edge,u)
    # TBD
end 

function current_tran!(meas, u, sys)
    params=data(sys)
    params=sys.physics.data
    U=reshape(u,sys)
    dx_end = sys.grid.coord[1,end] - sys.grid.coord[1,end-1]
    dphi_end = U[1, end] - U[1, end-1]
    dphiB=params.eps*(dphi_end/dx_end)
    Qb= - integrate(sys,reaction!,U) # \int n^F            
    Qs= params.z*(1-2*U[params.ibc,1])
    meas[1] = -Qb[1] - dphiB - Qs
    meas[2] = -Qb[1]
    meas[3] = -dphiB
    meas[4] = -Qs
end            

function current_stdy!(meas, u, sys)
    meas[1] = 0.0
end            

function main(;n=40,Plotter=nothing,voltmm=false,verbose=false,dense=false, EIS=false, pyplot=false)
    
    h=2.0/convert(Float64,n)
    X=VoronoiFVM.geomspace(0.0,1.0 ,1e-8,1e-1)
    grid=VoronoiFVM.Grid(X)

    data=Data()
    data.eps=1.0e-4
    data.z=-1
    data.iphi=1
    data.ic=2
    data.ibc=3

    data.DGR=-5 # changes
    data.kR=1.0e-0

    ic=data.ic
    ibc=data.ibc
    iphi=data.iphi
    
    breaction_list = [linear_adsorption!, exp_adsorption!, LoMA_adsorption!]


    data.names = [string(@show F)[length("Main.Example160_UnipolarDriftDiffusion1D.."):end] for F in breaction_list]

    physics=[VoronoiFVM.Physics(data=data,
                               num_species=3,
                               flux=sedanflux!,
                               reaction=reaction!,
                               storage=storage!,
                               bstorage=bstorage!,
                               breaction=ads
                               ) for (ii,ads) in enumerate(breaction_list)]
    if dense
        sys=VoronoiFVM.DenseSystem(grid,physics)
    else
        sys=[VoronoiFVM.SparseSystem(grid,physics[ii]) for (ii,ads) in enumerate(breaction_list)]
    end

    [enable_species!(sys[ii],iphi,[1]) for (ii,ads) in enumerate(breaction_list)]
    [enable_species!(sys[ii],ic,[1])   for (ii,ads) in enumerate(breaction_list)]

    [enable_boundary_species!(sys[ii],ibc,[1]) for (ii,ads) in enumerate(breaction_list)]

    phibc = -2.5
    ybc=0.2
    if EIS
        phibc=0.5*data.DGR
        ybc = 0.5
    end
    if voltmm
        phibc=0.0
        ybc =0.9934
    end 

    [boundary_dirichlet!(sys[ii],iphi,1,phibc) for (ii,ads) in enumerate(breaction_list)]
    [boundary_dirichlet!(sys[ii],iphi,2,0.0) for (ii,ads) in enumerate(breaction_list)]
    [boundary_dirichlet!(sys[ii],ic,2,0.5)   for (ii,ads) in enumerate(breaction_list)]

    
    inival=[unknowns(sys[ii])  for (ii,ads) in enumerate(breaction_list)]
    for (ii,ads) in enumerate(breaction_list)
        @views inival[ii][iphi,:].=0.0
        @views inival[ii][ic,:].=0.5
        @views inival[ii][ibc,1] =ybc
    end
    U = [unknowns(sys[ii]) for (ii,ads) in enumerate(breaction_list)]
    [U[ii].=inival[ii] for (ii,ads) in enumerate(breaction_list)]
    Ub = [[inival[ii][ibc,1]] for (ii,ads) in enumerate(breaction_list)]
    
    
    control=VoronoiFVM.NewtonControl()
    control.verbose=verbose
    u1=0
    # relaxation to steady state
    plot_solution(Plotter,sys,inival, Ub,[0])
    control.damp_initial=0.5
    T = zeros(1)
    t=0.0
    tstep=10^(-log10(data.kR)-6)# experimental1.0e-11
    tend=maximum([40.0,40*tstep])
    tstep_exp_coef = 1.08
    while t<tend
        t=t+tstep
        for (ii,ads) in enumerate(breaction_list)
            solve!(U[ii],inival[ii],sys[ii],control=control,tstep=tstep)
            inival[ii].=U[ii]
            if verbose
                @printf("time=%g\n",t)
            end
            append!(Ub[ii], Float64(U[ii][ibc,1]))
        end
        tstep*=tstep_exp_coef
        append!(T,t)
        plot_solution(Plotter,sys,U,Ub,T)
    end
    if voltmm
        delta=1.0e-4
                phimax=5.0 # V
        vrate=1.0e-4 # V/s
        tstep= phimax/120/vrate # s
        vplus=[zeros(0) for (ii,ads) in enumerate(breaction_list)]
        cdlplus=[ [zeros(0),zeros(0),zeros(0),zeros(0)] for (ii,ads) in enumerate(breaction_list)]
        cdl=zeros(4)
        curr_val=zeros(4)
        step_val=zeros(4)
        for dir in [1,-1, 1]
            @show dir
            phi=0.0
            for rid in [1,-1]
                @show rid
                if abs(phi)>phimax
                    phi = dir*phimax
                end
                while abs(phi)<=phimax && dir*phi >= 0
                    for (ii,ads) in enumerate(breaction_list)
                        
                        current_tran!(curr_val, inival[ii] ,sys[ii])
                        sys[ii].boundary_values[iphi,1]=phi
                        solve!(U[ii],inival[ii],sys[ii],control=control, tstep=tstep)
                        current_tran!(step_val, U[ii] ,sys[ii])
                        cdl = (step_val - curr_val)./tstep./vrate
                        inival[ii].=U[ii]

                        [append!(cdlplus[ii][kk],xx) for (kk, xx) in enumerate(cdl)]

                        append!(vplus[ii],phi)
                    end
                    phi+=2*dir*rid*vrate*tstep
              end
          end
        end
        if isplots(Plotter)
            Plots=Plotter
            p1=Plots.plot(grid=true)
            p2=Plots.plot(grid=true)
            p3=Plots.plot(grid=true)
            p4=Plots.plot(grid=true)
            names = ["total","bulk dens", "bulk tail","surface"]
            for (ii,ads) in enumerate(breaction_list)
                for (ll,pplot) in enumerate([p1,p2,p3,p4])
                    Plots.plot!(pplot,vplus[ii],cdlplus[ii][ll], title=names[ll], ylabel="I/rate [F/m^2]")
              end
            end
            p = Plots.plot(p1,p2,p3,p4,dpi=150)
            Plots.gui(p)
        end
        return cdlplus[1][4], vplus
    end
    
    if EIS
    
    # 
    function meas_stdy(meas, u) 
        return current_stdy!(meas, u, sys[1])
    end

    function meas_tran(meas, u) 
        return current_tran!(meas, u, sys[1])
    end

    [boundary_dirichlet!(sys[ii],iphi,1,data.DGR/2) for (ii,ads) in enumerate(breaction_list)]
    [boundary_dirichlet!(sys[ii],iphi,2,0.0) for (ii,ads) in enumerate(breaction_list)]
    [boundary_dirichlet!(sys[ii],ic,2,0.5)   for (ii,ads) in enumerate(breaction_list)]
    
    excited_spec=iphi
    excited_bc=1
    excited_bcval=phibc


    # Create impedance system
    # re-use steady state from testing
    isys=[VoronoiFVM.ImpedanceSystem(sys[ii],U[ii],excited_spec, excited_bc) for (ii,ads) in enumerate(breaction_list)]

    # Derivatives of measurement functionals
    # For the Julia magic behind this we need the measurement functionals
    # as mutating functions writing on vectors.
    dmeas_stdy=[measurement_derivative(sys[ii],meas_stdy,U[ii]) for (ii,ads) in enumerate(breaction_list)]
    dmeas_tran=[measurement_derivative(sys[ii],meas_tran,U[ii]) for (ii,ads) in enumerate(breaction_list)]


    
    # Impedance arrays
    z_timedomain=zeros(Complex{Float64},0)
    z_freqdomain=[zeros(Complex{Float64},0) for (ii,ads) in enumerate(breaction_list)]

    EIS_TDS=false # not implemented yet
    EIS_IS=true

    all_w=zeros(0)
    for (ii,ads) in enumerate(breaction_list)
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
                zfreq=freqdomain_impedance(isys[ii],w,U[ii],excited_spec,excited_bc,excited_bcval, dmeas_stdy[ii], dmeas_tran[ii])
                push!(z_freqdomain[ii],1.0/zfreq)
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

        for (ii,ads) in enumerate(breaction_list)
            if EIS_IS
                PyPlot.semilogx(all_w,positive_angle.(1.0/z_freqdomain[ii])',label="\$i\\omega\$")
            end
            if EIS_TDS
                PyPlot.semilogx(all_w,positive_angle.(1.0/z_timedomain)',label="\$\\frac{d}{dt}\$",color=:green)
            end
        end
        PyPlot.xlabel("\$\\omega\$")
        PyPlot.ylabel("\$\\phi\$")
        PyPlot.legend(loc="upper left")


        PyPlot.subplot(312)
        PyPlot.grid()
        for (ii,ads) in enumerate(breaction_list)
            if EIS_IS
                PyPlot.loglog(all_w,abs.(1.0/z_freqdomain[ii])',label="\$i\\omega\$")
            end
            if EIS_TDS
                PyPlot.loglog(all_w,abs.(1.0/z_timedomain)',label="\$\\frac{d}{dt}\$",color=:green)
            end
        end
        PyPlot.xlabel("\$\\omega\$")
        PyPlot.ylabel("a")
        PyPlot.legend(loc="lower left")

        
        PyPlot.subplot(313)
        PyPlot.grid()
        for (ii,ads) in enumerate(breaction_list)
            if EIS_IS
                PyPlot.plot(real(z_freqdomain[ii]),-imag(z_freqdomain[ii]),label="\$i\\omega\$")
            end
            if EIS_TDS
                #PyPlot.plot(real(z_timedomain),-imag(z_timedomain),label="\$\\frac{d}{dt}\$", color=:green)
            end
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
