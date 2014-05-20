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

    gnuplot_cmd = ""

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
                         "-gens", "100",
                         "-tris", "800",
                         "-psoiter", "1000",
                         "-checklimit", "150",
                         "-particles", "16",
                         img
                         ])
        gnuplot_cmd += "'%s' using 1:3 title '%s' with lines, " % (stats_file, root)
    subprocess.call(["gnuplot", "-e",
                     "load 'evogpupso.p'; set output '"+out_dir+"/graph-fitness.jpg'; plot " + gnuplot_cmd])