using Test

@time begin
    @time begin
        print("including ysz-new:")
        include("../src/ysz-new.jl")
    end

    @time begin
        print("       ysz-new s2:")
        @test YSZNew.run_new(test=true, voltammetry=true, sample=2) ≈ 0.4465812609631136
    end
    
    @time begin
        print("       ysz-new s5:")
        @test YSZNew.run_new(test=true, voltammetry=true, sample=5) ≈ 0.38298931399680425
    end

    @time begin
        print("      ysz-new s10:")
        @test YSZNew.run_new(test=true, voltammetry=true, sample=10) ≈ 0.35930102641606637
    end

    @time begin
        print(" ysz-new dlcap s2:")
        @test YSZNew.run_new(test=true, voltammetry=true, sample=2, dlcap=true) ≈ 0.26027444761018886
    end
    
    @time begin
        print(" ysz-new dlcap s5:")
        @test YSZNew.run_new(test=true, voltammetry=true, sample=5, dlcap=true) ≈ 0.13318328475918376
    end
    
    @time begin
        print("ysz-new dlcap s10:")
        @test YSZNew.run_new(test=true, voltammetry=true, sample=10, dlcap=true) ≈ 0.07253356390937356
    end
    
    print("          all:")
end
