using Printf
using Plots
using LinearAlgebra
using Kronecker

# uncomment to save data
using DataFrames
using CSV

function Condat_Vu(RR, RG, M, M_tran, ic, ϵ, α; debug = false, L = 1e4, verbose=true, min_runs = 0)
    # ϵ is the relative convergence tolerance. α is the resolvent parameter.  L is the definition of instability. A is evaluated forward. RB is the α resolvent of B.
    count = 0
    i0 = zeros(length(ic)*2, 1)
    i1 = zeros(length(ic)*2, 1)
    v0 = zeros(length(ic)*1, 1)
    v1 = zeros(length(ic)*1, 1)
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

# resolvent of inductor impedance
function J_L(N, α; L = 1, T = 0.02)
    diff = Array(Bidiagonal(vec(ones(1, N)), -vec(ones(1, N - 1)), :L))
    diff[1, end] = -1
    diff = N/T*diff # divide by dt
    S = L*diff 
    return inv(I + α*S)
end

# resolvent of capacitor admittance
function J_C(N, α; C = 1, T = 0.02)
    diff = Array(Bidiagonal(vec(ones(1, N)), -vec(ones(1, N - 1)), :L))
    diff[1, end] = -1
    diff = N/T*diff # divide by dt
    S = C*diff 
    return inv(I + α*S)
end

# rectified linear unit
function relu(x)
    return x > 0.0 ? x : 0.0
end

# resolvent for impedances, with input u = vp, evaluated at x
function RR(x, u, α, ϵ)
         n = length(x)
         N = div(n, 2)
         R = 1.0
         x1 = (x[N+1:end])./(1 + α*R)
         x0 = J_L(N, α, L = 0.001)*(x[1:N] .+ α*u)
         return vcat(x0, x1)
end

# resolvent for admittances, with input u = iq, evaluated at x
function RG(x, u, α, ϵ)
         N = length(x)
         return J_C(N, α, C = 0.01)*(x .- α*u)
end

# matrix M
function M(x)
        n = length(x)
        N = div(n, 2)
        return x[1:N] .- x[N+1:end]
end

# transpose of M
function M_tran(x)
        return vcat(x, -x)
end


function run_example()
    N = 200
    end_time = 0.02
    T = LinRange(0, end_time, N)
    vp = sin.(T*2*pi/end_time)
    iq = zeros(size(vp))
    ic = ones(N, 1)

    α = 0.05
    ϵ = 0.001

    z = Condat_Vu(x -> RR(x, vp, α, ϵ), x -> RG(x, iq, α, ϵ), M, M_tran, ic, ϵ, α, min_runs=0)

    i1 = z[1:N]
    i3 = z[N+1:2N]
    v2 = z[2N+1:3N]

    ip = -i1
    vq = v2 
    
    p1 = plot(T, vq, label = "v_q", ylabel = "V") 
    plot!(T, vp, label = "v_p", ylabel = "V", xlabel = "Time (s)", legend=:bottomleft)

    p2 = plot(T, iq, label = "i_q", ylabel = "A")
    plot!(T, ip, label = "i_p", ylabel = "A", xlabel = "Time (s)", legend=:bottomright)

    display(plot(p1, p2, layout = (2, 1)))

    d = DataFrame(t = T, vp = vp, vq = vq, iq = iq, ip = ip)
    CSV.write("bridge_rectifier.csv", d)
                
end

run_example()
