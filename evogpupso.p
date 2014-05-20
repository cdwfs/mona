unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set xlabel "Iterations (log)"
set ylabel "PSNR (dB)"
set xr [1:10000]
set yr [0:30]
set logscale x
set terminal jpeg size 800,600 font "Arial,12"
