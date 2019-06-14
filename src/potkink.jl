module PotKink

using Printf

using VoronoiFVM

# Allow plotting
if isinteractive()
    using PyPlot
end




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
    
    if plotgrid
        clf()
        fvmplot(grid)
        show()
        waitforbuttonpress()

    end

    Q=0.0

    function flux!(f,u,edge,data)
        f[1]=u[1]-u[2]
    end
    function storage!(f,u,node,data)
        f[1]=u[1]
    end

    # Define boundary reaction defining charge
    # Note that the term  is written on  the left hand side, therefore the - sign
    # Alternatively,  can put the charge
    # into the boundary reaction term.
    function breaction!(f,u,node,data)
        if node.region==3
            f[1]=-Q
        end
    end

    # Create physics
    physics=VoronoiFVM.Physics(
        flux=flux!,
        storage=storage!,
        breaction=breaction!
    )

    # Create system
    sys=VoronoiFVM.DenseSystem(grid,physics)

    #  put potential into both regions
    enable_species!(sys,1,[1,2])

    # Set boundary conditions
    sys.boundary_values[1,1]=1.0
    sys.boundary_values[1,2]=0.0
    sys.boundary_factors[1,1]=VoronoiFVM.Dirichlet
    sys.boundary_factors[1,2]=VoronoiFVM.Dirichlet


    
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

    # Solve and plot for several values of charge
    for q in [0.1,0.2,0.4,0.8,1.6]
        # surface charge at x=0
        
        if brea
            # Charge in reaction term
            Q=q
        else
            # Charge as boundary condition
            sys.boundary_values[1,3]=q
        end
        solve!(U,inival,sys, control=control)
        
        # Plot data
        if pyplot
            PyPlot.grid()
            plot(grid.coord[1,:],U[1,:],label=@sprintf("q=%.2f",q))
            PyPlot.legend(loc="upper right")
            pause(1.0e-10)
        end
    end
end


end 

