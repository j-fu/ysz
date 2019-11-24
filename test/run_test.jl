using Test

@time begin
    @time begin
        print("including ysz_experiments:")
        include("../examples/ysz_experiments.jl")
    end

    @time begin
        print("       ysz_experiments s2:")
        @test ysz_experiments.run_new(test=true, voltammetry=true, sample=2) ≈ 0.4465812609631136
    end
    
    @time begin
        print("       ysz_experiments s5:")
        @test ysz_experiments.run_new(test=true, voltammetry=true, sample=5) ≈ 0.38298931399680425
    end

    @time begin
        print("      ysz_experiments s10:")
        @test ysz_experiments.run_new(test=true, voltammetry=true, sample=10) ≈ 0.35930102641606637
    end

    @time begin
        print(" ysz_experiments dlcap s2:")
        @test ysz_experiments.run_new(test=true, voltammetry=true, sample=2, dlcap=true) ≈ 0.26027444761018886
    end
    
    @time begin
        print(" ysz_experiments dlcap s5:")
        @test ysz_experiments.run_new(test=true, voltammetry=true, sample=5, dlcap=true) ≈ 0.13318328475918376
    end
    
    @time begin
        print("ysz_experiments dlcap s10:")
        @test ysz_experiments.run_new(test=true, voltammetry=true, sample=10, dlcap=true) ≈ 0.07253356390937356
    end
    
    print("          all:")
end
