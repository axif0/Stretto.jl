module StrettoMakieExt

using Stretto
using Makie
using Piccolo: AbstractPulse

# pulse_spectrum is called fully-qualified to avoid ambiguity inside the ext.
function Stretto.plot_pulse_spectrum(pulse::AbstractPulse; n_samples::Int=1000, bandwidth_GHz=nothing, kwargs...)
    freqs, amplitudes = Stretto.pulse_spectrum(pulse; n_samples=n_samples)

    n_drives = size(amplitudes, 1)

    fig = Figure(size = (700, 400))
    ax = Axis(
        fig[1, 1];
        title = "Pulse Amplitude Spectrum",
        xlabel = "Frequency (GHz)",
        ylabel = "Normalised Amplitude (|FFT| / N)",
        yscale = log10,
        kwargs...,
    )

    colors = Makie.wong_colors()
    for i in 1:n_drives
        # Clamp to ε before log10 to avoid -Inf on zero-amplitude bins
        lines!(ax, freqs, max.(amplitudes[i, :], 1e-12);
            label = "Drive $i",
            color = colors[mod1(i, length(colors))],
            linewidth = 1.5,
        )
    end

    # Optional bandwidth limit reference line
    if !isnothing(bandwidth_GHz)
        vlines!(ax, [bandwidth_GHz];
            color = :red,
            linestyle = :dash,
            linewidth = 1.5,
            label = "Bandwidth limit",
        )
    end

    if n_drives > 1 || !isnothing(bandwidth_GHz)
        axislegend(ax; position = :rt)
    end

    return fig
end

end
