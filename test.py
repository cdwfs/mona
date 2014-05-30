import datetime
import os
import posixpath as path
import subprocess
import sys

inputs = [
    "Hanspeter-Pfister_large.jpg",
    "lena.jpg",
    "lisa.jpg",
    "Mondrian.jpg",
    "monet-parliament.jpg",
    "moonman.jpg",
    "picasso_three_musicians_moma.jpg",
    "slisa.jpg",
]

if __name__ == "__main__":
    #TODO: build evogpupso

    out_dir = "test-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.mkdir(out_dir)

    gnuplot_fitness_file = path.join(out_dir, "fitness.p")
    fitness_p = open(gnuplot_fitness_file, "w")
    fitness_p.write("""\
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set key outside right top
set xlabel "Generation"
set ylabel "PSNR (dB)"
set xr [1:10000]
set yr [0:40]
set logscale x
set terminal jpeg size 800,600 font "Arial,12"
set output "graph-fitness.jpg"
plot """)

    gnuplot_time_file = path.join(out_dir, "time.p")
    time_p = open(gnuplot_time_file, "w")
    time_p.write("""\
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set key outside right top
set xlabel "Generation"
set ylabel "Time (seconds)"
set xr [1:10000]
set yr [0:*]
set logscale x
set terminal jpeg size 800,600 font "Arial,12"
set output "graph-time.jpg"
plot """)

    for img in inputs:
        root,suffix = os.path.splitext(img)

        out_file = path.join(out_dir, root+"-out.png")
        stats_file = path.join(out_dir, root+"-out.dat")
        temp_dir = path.join(out_dir, root)
        os.mkdir(temp_dir)
        subprocess.call(["evogpupso",
                         "-out", out_file,
                         "-stats", stats_file,
                         "-temp", temp_dir,
                         "-scale", "2",
                         "-gens", "2000",
                         "-tris", "800",
                         "-psoiter", "1000",
                         "-checklimit", "150",
                         "-particles", "16",
                         img
                         ])
        fitness_p.write("'%s' using 1:3 title '%s' with lines,\\\n" % (os.path.split(stats_file)[1], root))
        time_p.write("'%s' using 1:2 title '%s' with lines,\\\n" % (os.path.split(stats_file)[1], root))

    fitness_p.close()
    time_p.close()
    os.chdir(out_dir)
    subprocess.call([ "gnuplot", "-e", "load '%s'" % os.path.split(gnuplot_fitness_file)[1] ])
    subprocess.call([ "gnuplot", "-e", "load '%s'" % os.path.split(gnuplot_time_file)[1] ])