from random import shuffle
import subprocess
from tqdm import tqdm
import sys

if __name__ == "__main__":
    N = 4
    L10 = 3
    L2 = 3
    A = {((i % 9) * (10 ** (i // 9))) for i in range(1, L10 + 1)}
    B = {(2 ** i) for i in range(1, L2 + 1)}

    elements = sorted(list(A | B))

    d = [(bn, bs) for bn in elements for bs in elements]

    shuffle(d)

    matmul_types = ["2d-1c-1s", "3d-1c-1s", "3d-2c-1s", "4d-1c-1s", "4d-1c-2s"]

    def run(args):
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        exit_code = proc.wait()
        if exit_code:
            print(f"{exit_code=}", file=sys.stderr)
        return not bool(exit_code)

    def read_mem(path, exts):
        for ext in exts:
            with subprocess.Popen(["/home/florian/go/bin/pprof","-top", path + ext + ".prof"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout as p:
                p.readline() # ignore the first line
                mem = p.readline().decode().split(" ")[-2]
            with open(path + ext + ".tsv", "a") as f:
                f.write(str(mem) + "\n")
            # delete.prof?
    
    t1 = tqdm(total=len(d)**2*4)
    for i, (bn, bs) in enumerate(d):
        t1.set_description(f"{bn=}, {bs=}")
        main_call = [
            "uv", "run", "performance_measurements/bst_perf_test.py",
            "-p", "-bn", str(bn), "-bs", str(bs)
        ]
        for matmul_type in matmul_types:
            main_call += ["-t", matmul_type]
            for sparse_matmul_flag in {0, 1}:
                for sparse_rhs_flag in {0, 1}:
                    t2 = tqdm(total=N*2, leave=False)
                    cur_flags = ["--sparse-matmul" if sparse_matmul_flag else "", "--sparse-rhs" if sparse_rhs_flag else ""]
                    cur_flags = [flag for flag in cur_flags if flag]

                    t2.set_description(f"rhs={'sparse' if sparse_rhs_flag else 'dense'}, matmul={'sparse' if sparse_matmul_flag else 'dense'}")

                    path_components = [
                        f"bn{bn}",
                        f"bs{bs}",
                        f"t{matmul_type.replace('-','')}",
                        "sparse_rhs" if sparse_rhs_flag else "",
                        "sparse_matmul" if sparse_matmul_flag else "",
                    ]

                    path = "_".join([x for x in path_components if x])

                    ok = run(main_call + cur_flags + ["-nop"])
                    read_mem(path, ["_baseline"])
                    t2.update()
                    if ok:
                        for j in range(N-1):
                            run(main_call + cur_flags + ["-nop", "-s", str(i+j)])
                            read_mem(path, ["_baseline"])
                            t2.update()
                    
                    ok = run(main_call + cur_flags)
                    read_mem(path, ["_comp", "_bench"])
                    t2.update()
                    if ok:
                        for j in range(N-1):
                            run(main_call + cur_flags + ["-s", str(i+j)])
                            read_mem(path, ["_comp", "_bench"])
                            t2.update()

                    t1.update()

            main_call = main_call[:-2]
