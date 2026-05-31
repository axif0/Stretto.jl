"""
    default_post_process() -> Vector{Function}

Return the ordered list of post-processing transforms applied after a block
solve. Each entry has signature `(block_result, ctx::PostProcessContext) -> block_result'`.

Substrate: empty list (no post-processing). Strategies may override with
their own `post_process::Vector{Function}` via the CompilationStrategy struct;
this seam is the codebase-wide default consulted only when no strategy is
selected.
"""
default_post_process() = _DEFAULT_POST_PROCESS[]()

_substrate_default_post_process() = Function[]

"""
    pulse_spectrum(pulse; n_samples=1000)

Sample the pulse and compute the one-sided amplitude spectrum of each drive
channel using a real-to-complex FFT.

Pulse duration is assumed to be in nanoseconds, so frequencies are returned
in GHz via `freq_GHz = rfftfreq(n, 1/dt_ns)`.

Returns `(freqs_GHz, amplitudes)` where:
- `freqs_GHz` is a vector of length `n_samples ÷ 2 + 1` (one-sided)
- `amplitudes` is a matrix of shape `(n_drives, n_frequencies)` containing
  the normalised amplitude spectrum `|FFT| / n` for each drive channel.
"""
function pulse_spectrum(pulse::AbstractPulse; n_samples::Int = 1000)
    # sample(pulse, n::Int) -> (controls::Matrix, times::Vector)
    controls, times = Piccolo.sample(pulse, n_samples)
    n = length(times)
    dt_ns = times[2] - times[1]

    # One-sided frequencies in GHz (pulse duration in ns → freq in GHz)
    freqs = rfftfreq(n, 1 / dt_ns)

    # One-sided amplitude spectrum, normalised by n so the result is independent
    # of sample count.  rfft operates along the last (time) dimension.
    amplitudes = abs.(rfft(controls, 2)) ./ n

    return freqs, amplitudes
end

const _DEFAULT_POST_PROCESS = Ref{Any}(_substrate_default_post_process)

"""
    set_default_post_process!(f)

Install `f` as the substrate post-process builder. `f` must have signature
`() -> Vector{Function}`.
"""
set_default_post_process!(f) = (_DEFAULT_POST_PROCESS[] = f)

@testitem "pulse_spectrum — returns peaks at expected frequencies" begin
    using Stretto
    using Piccolo: ZeroOrderPulse

    # 500 ns window, 5000 samples → dt = 0.1 ns → Nyquist = 5 GHz.
    # Drive frequencies (1 GHz and 3 GHz) are well below Nyquist so the
    # FFT peaks land cleanly on the expected bins.
    times = collect(range(0.0, 500.0, length = 5000))
    # Drive 1: 1 GHz sine wave
    # Drive 2: 3 GHz sine wave
    u1 = sin.(2π * 1.0 .* times)
    u2 = sin.(2π * 3.0 .* times)
    u = vcat(u1', u2')

    pulse = ZeroOrderPulse(u, times)
    freqs, amplitudes = pulse_spectrum(pulse; n_samples=5000)

    # Check shape
    @test size(amplitudes, 1) == 2
    @test size(amplitudes, 2) == length(freqs)

    # Find peak frequencies
    peak_idx1 = argmax(amplitudes[1, :])
    peak_idx2 = argmax(amplitudes[2, :])

    @test freqs[peak_idx1] ≈ 1.0 atol=0.02
    @test freqs[peak_idx2] ≈ 3.0 atol=0.02
end

@testitem "PostProcessContext — basic construction" begin
    using Stretto
    using Piccolo: UnitaryTrajectory, CubicSplinePulse, QuantumSystem, SplinePulseProblem

    σz = ComplexF64[1 0; 0 -1];
    σx = ComplexF64[0 1; 1 0]
    sys = QuantumSystem(σz, [σx], [1.0])
    times = collect(range(0.0, 10.0, length = 5))
    pulse = CubicSplinePulse(
        zeros(1, 5),
        zeros(1, 5),
        times;
        initial_value = zeros(1),
        final_value = zeros(1),
    )
    qtraj = UnitaryTrajectory(sys, pulse, ComplexF64[1 0; 0 1])
    problem = SplinePulseProblem(qtraj; Q = 100.0)

    device = HeronR3()
    circuit = GateCircuit([GateOp(:H, (1,))], 1)

    ctx = Stretto.PostProcessContext(circuit, device, qtraj, problem)
    @test ctx.circuit === circuit
    @test ctx.device === device
    @test ctx.qtraj === qtraj
    @test ctx.problem === problem
end

@testitem "default_post_process — substrate is empty list" begin
    using Stretto

    pp = Stretto.default_post_process()
    @test pp isa Vector
    @test isempty(pp)
end
