using Printf
using Plots
using LinearAlgebra
using Kronecker

# uncomment to save data
using DataFrames
using CSV

# this version of the example leaves iq as an exogenous input
#

function Condat_Vu(RR, RG, M, M_tran, ic, ϵ, α; debug = false, L = 1e4, verbose=true, min_runs = 0)
    # ϵ is the relative convergence tolerance. α is the resolvent parameter.  L is the definition of instability. A is evaluated forward. RB is the α resolvent of B.
    count = 0
    i0 = zeros(length(ic)*3, 1)
    i1 = zeros(length(ic)*3, 1)
    v0 = zeros(length(ic)*2, 1)
    v1 = zeros(length(ic)*2, 1)
    while true
        count += 1
        v0 = copy(v1)
        i0 = copy(i1)

        i1 = RR(i0 + α*M_tran(v0))
        v1 = RG(v0 + α*M(-2.0*i1 + i0))


        if count < min_runs
            continue
        end
        if maximum(abs.(i0 - i1)) < ϵ #/maximum(abs.(y0)) < ϵ
            @printf("Terminating on i convergence %f, iteration %d\n", maximum(abs.(i0 - i1)), count)
            break
        elseif maximum(abs.(v0 - v1)) < ϵ #/maximum(abs.(y0)) < ϵ
            @printf("Terminating on v convergence %f, iteration %d\n", maximum(abs.(v0 - v1)), count)
            break
        elseif maximum(abs.(i0)) > L
            throw(UnstableException(string(count)))
        end

        if count%101 == 0 && verbose
            @printf("Count: %d Absolute tolerance: %0.6f\n", count,  maximum(abs.(i0 - i1)))
        end

        if count%101 == 0 && debug
        end
    end
    return vcat(i1, v1)
end

# resolvent of RC circuit impedance
function R_RC(N, α; R = 1, C = 1, T = 0.02)
    # differentiate q_c to give i_c
    diff = Array(Bidiagonal(vec(ones(1, N)), -vec(ones(1, N - 1)), :L))
    diff[1, end] = -1
    diff = N/T*diff # divide by dt
    S = (1/R)*I + C*diff 
    return inv(α*I + S)*S
end

# rectified linear unit
function relu(x)
    return x > 0.0 ? x : 0.0
end

# resolvent for impedances, with input u = vp, evaluated at x
function RR(x, u, α, ϵ)
         n = length(x)
         N = div(n, 3)
         v = vcat(-u./24, u./24) # apply input transformation 
         x1 = relu.(x[N+1:end] .+ α*v) 
         x0 = R_RC(N, α, R = 1.0e3, C = 10.0e-6)*x[1:N]
         return vcat(x0, x1)
end

# resolvent for admittances, with input u = iq, evaluated at x
function RG(x, u, α, ϵ)
         v = zeros(size(x))
         v = vcat(u, u)
         -relu.(-x .- α*v)
end

# matrix M
function M(x)
        n = length(x)
        N = div(n, 3)
        return vcat(-x[1:N] .+ x[2*N+1:end], -x[1:N] .+ x[N+1:2*N])
end

# transpose of M
function M_tran(x)
        n = length(x)
        N = div(n, 2)
        return vcat(-x[1:N] .- x[N+1:end], x[N+1:end], x[1:N])
end


function run_example()
    N = 200
    end_time = 0.02
    T = LinRange(0, end_time, N)
    vp = 240.0*sin.(T*2*pi/end_time)
    iq = -0.005*ones(size(vp))
    ic = ones(N, 1)

    α = 0.05
    ϵ = 0.001

    z = Condat_Vu(x -> RR(x, vp, α, ϵ), x -> RG(x, iq, α, ϵ), M, M_tran, ic, ϵ, α, min_runs=20000)
    i0 = z[1:N]
    i1 = z[N+1:2N]
    i2 = z[2N+1:3N]
    v3 = z[3N+1:4N]
    v4 = z[4N+1:5N]

    ip = (i1 - i2)
    vq = -v3 - v4 
    
    p1 = plot(T, vq, label = "v_q", ylabel = "V") 
    plot!(T, vp./24, label = "v_p/24", ylabel = "V", xlabel = "Time (s)", legend=:bottomleft)

    p2 = plot(T, iq, label = "i_q", ylabel = "A")
    plot!(T, ip, label = "24i_p", ylabel = "A", xlabel = "Time (s)", legend=:bottomright)

    display(plot(p1, p2, layout = (2, 1)))

    d = DataFrame(t = T, vp = vp./24, vq = vq, iq = iq, ip = ip)
    CSV.write("bridge_rectifier.csv", d)
                
end

run_example()
