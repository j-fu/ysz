module PotKink

using Printf

using VoronoiFVM

using PyPlot


function main(;nref=0,pyplot=false,plotgrid=false,verbose=false, dense=false, brea=false)
    
    # Create grid in (-1,1) refined around 0
    hmax=0.2/2.0^nref
    hmin=0.05/2.0^nref
    X1=VoronoiFVM.geomspace(-1.0,0.0, hmax,hmin)
    X2=VoronoiFVM.geomspace(0.0,1.0, hmin,hmax)
    X=glue(X1,X2)
    grid=VoronoiFVM.Grid(X)
    


    # Edit default region numbers:
    #   additional boundary region 3 at 0.0
    bfacemask!(grid, [0.0],[0.0],3)
    # Material 1 left of 0
    cellmask!(grid, [-1.0],[0.0],1)
    # Material 2 right of 0
    cellmask!(grid, [0.0],[1.0],2)

    regions1=[1,2]
    regions2=[1,2]
    subgrid1=subgrid(grid,regions1)
    subgrid2=subgrid(grid,regions2)
    
    if plotgrid
        clf()
        VoronoiFVM.plot(PyPLot,grid)
        show()
        waitforbuttonpress()

    end

    Q=0.0

    function flux!(f,u,edge,data)
        uk=viewK(edge,u)
        ul=viewL(edge,u)
        f[1]=uk[1]-ul[1]
        f[2]=uk[2]-ul[2]
    end
    function storage!(f,u,node,data)
        f[1]=u[1]
        f[2]=u[2]
    end

    # Define boundary reaction defining charge
    # Note that the term  is written on  the left hand side, therefore the - sign
    # Alternatively,  can put the charge
    # into the boundary reaction term.
    function breaction!(f,u,node,data)
        if node.region==3
            f[2]=-Q
            f[3]=u[3]-3
        end
    end
    
    function bstorage!(f,u,node,data)
        if node.region==3
            f[3]=u[3]
        end
    end
    

    # Create physics
    physics=VoronoiFVM.Physics(
        num_species=3,
        flux=flux!,
        storage=storage!,
        breaction=breaction!,
        bstorage=bstorage!
    )

    # Create system
    sys=VoronoiFVM.DenseSystem(grid,physics)

    #  put potential into both regions
    enable_species!(sys,1,regions1)
    enable_species!(sys,2,regions2)
    enable_boundary_species!(sys,3,[3])

    # Set boundary conditions
    sys.boundary_values[1,1]=1.0
    sys.boundary_values[1,2]=0.0
    sys.boundary_factors[1,1]=VoronoiFVM.Dirichlet
    sys.boundary_factors[1,2]=VoronoiFVM.Dirichlet
    
    #sys.boundary_values[2,2]=2.0
    #sys.boundary_factors[2,2]=VoronoiFVM.Dirichlet
    #sys.boundary_values[2,3]=3.0
    #sys.boundary_factors[2,3]=VoronoiFVM.Dirichlet
    
    


    
    # Create a solution array
    inival=unknowns(sys)
    U=unknowns(sys)
    inival.=0

    # Create solver control info
    control=VoronoiFVM.NewtonControl()
    control.verbose=verbose
    if pyplot
        PyPlot.clf()
    end

    surfgrid = subgrid(grid,[3],boundary=true)
    @show surfgrid
    # Solve and plot for several values of charge
    for q in [3.0] #[0.0, 0.1,0.2,0.4,0.8,1.6]
        # surface charge at x=0
        
        if brea
            # Charge in reaction term
            Q=q
        else
            # Charge as boundary condition
            sys.boundary_values[1,3]=q
            sys.boundary_factors[1,3]=VoronoiFVM.Dirichlet
        end
        solve!(U,inival,sys, control=control)
        
        @show view(U, surfgrid)[3]
            
        # Plot data
        if pyplot
            PyPlot.grid()            
            VoronoiFVM.plot(PyPlot,subgrid1, U[1,:],label="spec1", color=(0.5,0,0))
            VoronoiFVM.plot(PyPlot,subgrid2, U[2,:],label="spec2", color=(0.0,0.5,0),clear=false)
            #plot(X, U[1,:],label="spec1", color=(0.5,0,0))
            #plot(X, U[2,:],label="spec2", color=(0.0,0.5,0))
            PyPlot.legend(loc="upper right")
            pause(1.0e-10)
        end
    end
end


end 

