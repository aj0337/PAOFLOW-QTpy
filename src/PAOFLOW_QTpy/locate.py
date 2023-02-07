def locate_extrema(nfft, fft_grid, eps):
    jl = 0
    ju = nfft + 1
    while (ju - jl > 0):
        jm = (ju + jl) / 2
        if (fft_grid[nfft - 1] > fft_grid[0] and eps > fft_grid[jm]):
            jl = jm
        else:
            ju = jm
    is_extrema = jl
    return is_extrema
