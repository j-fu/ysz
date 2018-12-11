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
    nus::Float64    # ratio of immobile ions on surface, \nu [1]
    ML::Float64   # averaged molar mass [kg]
    zL::Float64   # average charge number [1]
    DD::Float64   # diffusion coefficient [m^2/s]
    DDs::Float64   # surface adsorption coefficient 
    y0::Float64   # electroneutral value [1]
    dPsi::Float64 # difference of gibbs free energy
    dPsiR::Float64 # difference of gibbs free energy of electrochemical reaction
    areaL::Float64 # area of one FCC cell, a_L [m^2]
    R0::Float64 # exhange current density [A/m^2]
    pO::Float64 # O2 partial pressure [bar]
    #
    e0::Float64  
    eps0::Float64
    kB::Float64  
    N_A::Float64 
    zA::Float64  
    mO::Float64  
    mZr::Float64 
    mY::Float64 
    m_par::Float64
    ms_par::Float64
    #
    YSZParameters()= YSZParameters( new())
end

function YSZParameters(this)
    TwoPointFluxFVM.PhysicsBase(this,2)
    this.num_bspecies=[ 1, 0]
    this.chi=1.e1
    this.T=1073
    this.x_frac=0.2
    this.vL=3.35e-29
    this.areaL=(this.vL)^0.6666
    this.nu=0.9
    this.nus=0.9
    this.DD=1.0e-15
    this.DDs=1-3#.e3
    this.dPsi=-1.0e5
    this.dPsiR=-1.0
    this.R0=1.0e-3
    this.pO=1.0
    #
    this.e0   = 1.602176565e-19  #  [C]
    this.eps0 = 8.85418781762e-12 #  [As/(Vm)]
    this.kB   = 1.3806488e-23  #  [J/K]
    this.N_A  = 6.02214129e23  #  [#/mol]
    this.zA  = -2;
    this.mO  = 16/1000/this.N_A  #[kg/#]
    this.mZr = 91.22/1000/this.N_A #  [kg/#]
    this.mY  = 88.91/1000/this.N_A #  [kg/#]
    this.m_par = 2
    this.ms_par = this.m_par
    this.zL  = 4*(1-this.x_frac)/(1+this.x_frac) + 3*2*this.x_frac/(1+this.x_frac) - 2*this.m_par*this.nu
    this.y0  = -this.zL/(this.zA*this.m_par*(1-this.nu))
    this.ML  = (1-this.x_frac)/(1+this.x_frac)*this.mZr + 2*this.x_frac/(1+this.x_frac)*this.mY + this.m_par*this.nu*this.mO
    # this.ML=1.77e-25
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

function flux!(this::YSZParameters,f,uk,ul)
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
    f[iy]=(1+this.mO*this.m_par*(1.0-this.nu)*0.5*(uk[iy]+ul[iy])/this.ML)*this.DD*this.kB/this.mO*(bm*uk[iy]-bp*ul[iy]) # prefactor checked
end

function storage!(this::YSZParameters, f,u)
    f[iphi]=0
    f[iy]=this.mO*this.m_par*(1.0-this.nu)*u[iy]/this.vL
end

function reaction!(this::YSZParameters, f,u)
    # source term for the Poisson equation, beware of the sign
    f[iphi]=-(this.e0/this.vL)*(this.zA*u[iy]*this.m_par*(1-this.nu) + this.zL)
    f[iy]=0
end

function electroreaction(this::YSZParameters, bu)
    if this.R0 > 0
      eR = this.R0*((exp(this.dPsiR)*(bu[1]/(1-bu[1]))^0.5*(this.pO)^-0.25 ) - exp(-this.dPsiR)*((bu[1]/(1-bu[1]))^-0.5*(this.pO)^0.25))
      #eR = this.R0/this.e0*this.mO*sinh(this.dPsiR/this.T/this.kB + 0.5*log(bu[1]) - 0.5*log(1-bu[1]) - 0.25*log(this.pO))
      #eR = 1e-0*((exp(1)*(bu[1]/(1-bu[1]))^0.5*(this.pO)^-0.25 ) - exp(-1)*((bu[1]/(1-bu[1]))^-0.5*(this.pO)^0.25))
    else
        eR=0
    end
end

function breaction!(this::YSZParameters,f,bf,u,bu)
    if  this.bregion==1
        electroR=electroreaction(this,bu)
        f[iy]= this.DDs*(
                        this.dPsi + this.kB*this.T/this.mO*(
                          log(u[iy]*(1-bu[1])) - log(bu[1]*(1-u[iy]))
                        )
              )
        # if bulk chem. pot. > surf. ch.p. then positive flux from bulk to surf
        # sign is negative bcs of the equation implementation
        bf[1]= electroR-this.DDs*( # adsorption term
                        this.dPsi + this.kB*this.T/this.mO*(
                          log(u[iy]*(1-bu[1])) - log(bu[1]*(1-u[iy]))
                        )
              ) 
        f[iphi]=0
    else
        f[iy]=0
        f[iphi]=0
    end
end

function breaction2!(this::YSZParameters,f,bf,u,bu)
  if  this.bregion==1
      f[iy]=(u[iy]-bu[1])
      bf[1]=(bu[1]-u[iy])
  else
      f[1]=0
      f[2]=0
  end
end

function bstorage!(this::YSZParameters,bf,bu)
    if  this.bregion==1
        bf[1]=this.mO*this.ms_par*(1.0-this.nu)/this.areaL*bu[1]
    else
        bf[1]=0
    end
end
# function flux1!(this::YSZParameters,f,uk,ul)
#     f[iphi]=this.eps0*(1+this.chi)*(uk[iphi]-ul[iphi])
#     muk=-log(1-uk[iy])
#     mul=-log(1-ul[iy])
#     bp,bm=fbernoulli_pm(2*(uk[iphi]-ul[iphi])+(muk-mul))
#     f[iy]=bm*uk[iy]-bp*ul[iy]
# end

function direct_capacitance(this::YSZParameters, bound)
    # Clemens' analytic solution
    PHI = collect(-bound:0.001:bound) # PHI = phi_B-phi_S
    #
    yB = -this.zL/this.zA/this.m_par/(1-this.nu);
    X  = yB/(1-yB)*exp.(this.zA*this.e0/this.kB/this.T*PHI)
    y  = X./(1.0.+X)
    #
    nF = this.e0/this.vL*(this.zL.+this.zA*this.m_par*(1-this.nu).*y)
    F  = sign.(PHI).*sqrt.(
          2*this.e0/this.vL/this.eps0/(1.0+this.chi).*(
            this.zL.*PHI .+ this.kB*this.T/this.e0*this.m_par*(1-this.nu)*log.(
              (1-yB).*(X.+1.0)
             )
           )
         );
    #
    Y  = yB/(1-yB)*exp.(this.dPsi*this.mO .+ this.zA*this.e0/this.kB/this.T.*PHI);
    #
    CS = this.zA^2*this.e0^2/this.kB/this.T*this.ms_par/this.areaL*(1-this.nus)*Y./(1.0.+Y).^2;
    CBL  = nF./F;
    return CBL, CS, Y, PHI
end

function run_open(;n=15, verbose=false ,pyplot=false, width=2.0e-9, voltametry=false, dlcap=false,voltrate=1, bound=0.3, sample=300)
    #
    h=width/convert(Float64,n)
    X=collect(0.0:h:width)
    #
    geom=TwoPointFluxFVM.Graph(X)
    #
    parameters=YSZParameters()
    #
    parameters.storage=storage!
    parameters.flux=flux!
    parameters.reaction=reaction!
    parameters.breaction=breaction!
    parameters.bstorage=bstorage!
    #
    printfields(parameters)
    #print("weight ", parameters.mO*parameters.m_par*(1.0-parameters.nu)/parameters.vL,"\n")
    #
    sys=TwoPointFluxFVM.System(geom,parameters)
    #
    sys.boundary_values[iphi,1]=1.0e-0
    sys.boundary_values[iphi,2]=0.0e-3
    #
    sys.boundary_factors[iphi,1]=TwoPointFluxFVM.Dirichlet
    sys.boundary_factors[iphi,2]=TwoPointFluxFVM.Dirichlet
    #
    sys.boundary_values[iy,2]=parameters.y0
    sys.boundary_factors[iy,2]=TwoPointFluxFVM.Dirichlet
    #
    inival=unknowns(sys)
    inival.=0.0
    #
    inival_bulk=bulk_unknowns(sys,inival)
    for inode=1:size(inival_bulk,2)
        #inival_bulk[iphi,inode]=0.0e-3
        inival_bulk[iy,inode]= parameters.y0
    end
    inival_boundary = boundary_unknowns(sys,inival,1)
    inival_boundary[1]= parameters.y0
    #
    control=TwoPointFluxFVM.NewtonControl()
    control.verbose=verbose
    control.tol_linear=1.0e-4
    control.tol_relative=1.0e-5
    #control.tol_absolute=1.0e-4
    #control.max_iterations=3
    control.max_lureuse=0
    control.damp_initial=0.01
    control.damp_growth=2
    time=0.0
    if (!voltametry && !dlcap)
        time_range=zeros(0)
        istep=0
        Ub=zeros(0)
        tend=1.0e-4
        tstep=1.0e-7
        append!(time_range,time)
        append!(Ub,inival_boundary[1])
        #
        while time<tend
            time=time+tstep
            U=solve(sys,inival,control=control,tstep=tstep)
            inival.=U
            if verbose
                @printf("time=%g\n",time)
            end
            U_bulk=bulk_unknowns(sys,U)
            U_bound=boundary_unknowns(sys,U,1)
            append!(time_range,time)
            append!(Ub,U_bound[1,1])
            #
            if pyplot && istep%10 == 0
                @printf("max1=%g max2=%g maxb=%g\n",maximum(U_bulk[1,:]),maximum(U_bulk[2,:]),maximum(U_bound))
                PyPlot.clf()
                subplot(211)
                plot(X,U_bulk[1,:],label="spec1")
                plot(X,U_bulk[2,:],label="spec2")
                PyPlot.legend(loc="best")
                PyPlot.grid()
                subplot(212)
                plot(time_range,Ub,label="U_b")
                PyPlot.legend(loc="best")
                PyPlot.grid()
                pause(1.0e-10)
                #print(U_bound)
            end
            tstep*=1.05
        end
    #
    # voltametry
    #
    elseif voltametry
        istep=0
        phi=0
        Ub=zeros(0)
        phi_range=zeros(0)
        phi_range_full=zeros(0)
        Is_range=zeros(0)
        Ib_range=zeros(0)
        Ibb_range=zeros(0)
        r_range=zeros(0)
        print("calculating linear potential sweep\n")
        hc_count = 0
        dtstep=1e-4
        tstep=1/voltrate/sample
        dir=1
        while hc_count < 3
            if (phi <= -bound || phi >= bound) 
                dir*=(-1)
                phi+=tstep*dir*voltrate
                hc_count+=1
                print("cycle: ", hc_count,"\n")
            end
            # tstep to potential phi
            sys.boundary_values[iphi,1]=phi
            U=solve(sys,inival,control=control,tstep=tstep)
            inival.=U
            Qb=integrate(sys,reaction!,U) # - \int n^F
            dphiB=parameters.eps0*(1+parameters.chi)*(0 - bulk_unknowns(sys,U)[end][1])/h 
            y_bound=boundary_unknowns(sys,U,1)
            Qs= -(parameters.e0/parameters.areaL)*parameters.zA*y_bound*parameters.ms_par*(1-parameters.nus) # - n^F_s
            # dtstep to potential phi + voltrate*dtstep
            sys.boundary_values[iphi,1]=phi+voltrate*dtstep
            Ud=solve(sys,U,control=control,tstep=dtstep)
            Qbd=integrate(sys,reaction!,Ud)
            dphiBd = parameters.eps0*(1+parameters.chi)*(0 - bulk_unknowns(sys,Ud)[end][1])/h
            yd_bound=boundary_unknowns(sys,Ud,1)
            Qsd= -(parameters.e0/parameters.areaL)*parameters.zA*yd_bound*parameters.ms_par*(1-parameters.nus)
            # time derivatives
            dphiBdt = (-dphiB + dphiBd)/dtstep
            Ibb =- dphiBdt
            Ib=(Qbd[iphi] - Qb[iphi])/dtstep #- dphiBdt
            Is=(Qsd[1] - Qs[1])/dtstep
            # reaction average
            reac = electroreaction(parameters, y_bound)
            reacd = electroreaction(parameters, yd_bound)
            Ir=0.5*(reac + reacd)
            #
            if verbose
                @printf("time=%g\n",time)
            end
            U_bulk=bulk_unknowns(sys,U)
            U_bound=boundary_unknowns(sys,U,1)
            append!(Ub,U_bound[1,1])
            append!(phi_range_full,phi)
            # forget the initialization
            if hc_count > 0
                append!(phi_range,phi)
                append!(Is_range,Is)
                append!(Ib_range,Ib)
                append!(Ibb_range,Ibb)
                append!(r_range, Ir)
            end
            istep+=1
            #
            if pyplot && istep%10 == 0
                #@printf("dphiB=%g\n", dphiB)
                PyPlot.clf()
                subplot(211)
                plot(X,U_bulk[1,:],label="spec1")
                plot(X,U_bulk[2,:],label="spec2")
                PyPlot.legend(loc="best")
                PyPlot.grid()
                subplot(212)
                plot(collect(1:istep),Ub,label="U_b")
                plot(collect(1:istep),phi_range_full,label="phi")
                PyPlot.legend(loc="best")
                PyPlot.grid()
                pause(1.0e-10)
            end
            phi+=tstep*dir*voltrate
        end
        PyPlot.clf()
        subplot(221)
        plot(phi_range, Ib_range ,label="bulk")
        plot(phi_range, Ibb_range ,label="bulk_grad")
        plot(phi_range, Is_range ,label="surf")
        plot(phi_range, r_range ,label="reac")
        PyPlot.legend(loc="best")
        PyPlot.grid()
        subplot(222)
        plot(phi_range, Is_range + Ib_range + r_range + Ibb_range ,label="spec1")
        PyPlot.legend(loc="best")
        PyPlot.grid()
        subplot(223)
        #plot(phi_range, r_range ,label="spec1")
        PyPlot.legend(loc="best")
        PyPlot.grid()
        subplot(224)
        #plot(phi_range, r_range ,label="spec1")
        PyPlot.legend(loc="best")
        PyPlot.grid()
    elseif dlcap 
        parameters.R0=0;
        print("calculating double layer capacitance\n")
        for inode=1:size(inival,2)
            inival[iphi,inode]=0
            inival[iy,inode]=parameters.y0
        end
        sys.boundary_values[iphi,1]=0
        dphi=5.0e-3
        delta=1.0e-5
        phimax=bound#0.55
        v=zeros(0)
        cdl=zeros(0)
        cb=zeros(0)
        si=zeros(0)
        ys=zeros(0)
        for dir in [1,-1] # direction switch, neat...
            sol=copy(inival)
            phi=0.0
            while phi<phimax
                sys.boundary_values[iphi,1]=dir*phi
                sol=solve(sys,sol,control=control)
                Q=integrate(sys,reaction!,sol)
                y_bound=boundary_unknowns(sys,sol,1)
                Qs= -(parameters.e0/parameters.areaL)*parameters.zA*y_bound*parameters.ms_par*(1-parameters.nus)
                #print("Qs ", Qs,"\n")
                sys.boundary_values[iphi,1]=dir*phi+delta
                sol=solve(sys,sol,control=control)
                Qdelta=integrate(sys,reaction!,sol)
                yd_bound=boundary_unknowns(sys,sol,1)
                Qsd= -(parameters.e0/parameters.areaL)*parameters.zA*yd_bound*parameters.ms_par*(1-parameters.nus)
                c=(Qdelta[iphi] - Q[iphi])/delta
                s=(Qsd[1] - Qs[1])/delta
                cdl=(Qdelta[iphi] - Q[iphi] +Qsd[1]-Qs[1])/delta
                if dir==1
                    add=append!
                else
                    add=prepend!
                end
                add(v,dir*phi)
                add(cb,c)
                add(si,s)
                add(ys,y_bound)
              if true
                  #@printf("max1=%g max2=%g maxb=%g\n",maximum(U_bulk[1,:]),maximum(U_bulk[2,:]),maximum(U_bound))
                  U_bulk=bulk_unknowns(sys,sol)
                  PyPlot.clf()
                  subplot(211)
                  plot(X,U_bulk[1,:],label="phi")
                  plot(X,U_bulk[2,:],label="y")
                  PyPlot.legend(loc="best")
                  PyPlot.grid()
                  subplot(212)
                  plot(v,ys,label="FVM y_s")
                  PyPlot.legend(loc="best")
                  PyPlot.grid()
                  pause(1.0e-10)
                  #print(y_bound)
              end
                phi+=dphi
            end
        end
        if pyplot
            CBL, CS, Y, PHI= direct_capacitance(parameters,bound) # analytic solution by Clemens
            PyPlot.clf()
            subplot(411)
            PyPlot.plot(v,cb+si,color="g", label="fvm b+s")
            PyPlot.plot(PHI,reverse(CBL+CS),color="b", label="analytic b+s")
            PyPlot.grid()
            PyPlot.legend(loc="upper right")
            #
            subplot(412)
            PyPlot.plot(v, cb,color="g", label="FVM b")
            PyPlot.plot(PHI,reverse(CBL),color="b", label="analytic b")
            PyPlot.grid()
            PyPlot.legend(loc="upper right")
            #
            subplot(413)
            PyPlot.plot(v,si,color="g", label="FVM s")
            PyPlot.plot(PHI, reverse(CS),color="b", label="analytic s")
            PyPlot.grid()
            PyPlot.legend(loc="upper right")
            #
            subplot(414)
            PyPlot.plot(v,ys,color="g", label="fvm y_s")
            PyPlot.plot(PHI,Y,color="b", label="analytic y_s")
            PyPlot.grid()
            PyPlot.legend(loc="upper right")
        end
    end
end
