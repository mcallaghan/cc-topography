import os
writelines = True
collect_preamble = True
with open("main.tex", "r") as f:
    with open("main_sub.tex","w") as nf:
        with open("figures.tex","w") as tf:
            for l in f:
                if collect_preamble:
                    tf.write(l)
                    if "setcitestyle" in l:
                        collect_preamble = False
                        tf.write("\\begin{document}\n")
                        nf.write("\externaldocument{figures}\n")
                if writelines:
                    if "begin{figure}" in l:
                        tf.write(l)
                        writelines = False
                    else:
                        nf.write(l)
                else:
                    tf.write(l)
                    if "\end{figure}" in l:
                        writelines=True

            tf.write("\\end{document}\n")

os.system("pdflatex figures.tex")
