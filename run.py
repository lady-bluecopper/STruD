#!/usr/bin/env python
# coding: utf-8

import time
import configuration as conf
import simplicial_truss_decomposition as simp
import utils as ut


if __name__ == "__main__":
    dataset = ut.load_obj(conf.data_dir, conf.dataset)
    if conf.experiments[0] == 1:
        if conf.vary_q:
            for q in conf.qs:
                start_time = time.time()
                simp.baseline_impl(dataset, conf.num_files, q)
                with open('./stats', 'a') as stats_f:
                    total = time.time() - start_time
                    stats_f.write(f'{conf.dataset} imp {q} {total}\n')
        else:
            start_time = time.time()
            simp.baseline_impl(dataset, conf.num_files, conf.q)
            with open('./stats', 'a') as stats_f:
                total = time.time() - start_time
                stats_f.write(f'{conf.dataset} imp {conf.q} {total}\n')
    if conf.experiments[1] == 1:
        if conf.vary_q:
            for q in conf.qs:
                start_time = time.time()
                simp.baseline_expl(dataset, conf.num_files, q)
                with open('./stats', 'a') as stats_f:
                    total = time.time() - start_time
                    stats_f.write(f'{conf.dataset} exp {q} {total}\n')
        else:
            start_time = time.time()
            simp.baseline_expl(dataset, conf.num_files, conf.q)
            with open('./stats', 'a') as stats_f:
                total = time.time() - start_time
                stats_f.write(f'{conf.dataset} exp {conf.q} {total}\n')
    if conf.experiments[2] == 1:
        if conf.vary_q:
            for q in conf.qs:
                if conf.vary_n:
                    for n in conf.ns:
                        start_time = time.time()
                        simp.baseline_top_q(dataset, conf.num_files, n, q)
                        with open('./stats', 'a') as stats_f:
                            total = time.time() - start_time
                            stats_f.write(f'{conf.dataset} topk {n} {q} {total}\n')
                else:
                    start_time = time.time()
                    simp.baseline_top_q(dataset, conf.num_files, conf.n, q)
                    with open('./stats', 'a') as stats_f:
                        total = time.time() - start_time
                        stats_f.write(f'{conf.dataset} topk {conf.n} {q} {total}\n')
        else:
            if conf.vary_n:
                for n in conf.ns:
                    start_time = time.time()
                    simp.baseline_top_q(dataset, conf.num_files, n, conf.q)
                    with open('./stats', 'a') as stats_f:
                        total = time.time() - start_time
                        stats_f.write(f'{conf.dataset} topk {n} {conf.q} {total}\n')
            else:
                start_time = time.time()
                simp.baseline_top_q(dataset, conf.num_files, conf.n, conf.q)
                with open('./stats', 'a') as stats_f:
                    total = time.time() - start_time
                    stats_f.write(f'{conf.dataset} topk {conf.n} {conf.q} {total}\n')
    if conf.experiments[3] == 1:
        if conf.vary_q:
            for q in conf.qs:
                if conf.vary_k:
                    for k in conf.ks:
                        start_time = time.time()
                        simp.baseline_k_q(dataset, conf.num_files, k, conf.min_q, q)
                        with open('./stats', 'a') as stats_f:
                            total = time.time() - start_time
                            stats_f.write(f'{conf.dataset} ktruss {k} {conf.min_q} {q} {total}\n')
                else:
                    start_time = time.time()
                    simp.baseline_k_q(dataset, conf.num_files, conf.k, conf.min_q, q)
                    with open('./stats', 'a') as stats_f:
                        total = time.time() - start_time
                        stats_f.write(f'{conf.dataset} ktruss {conf.k} {conf.min_q} {q} {total}\n')
        else:
            if conf.vary_k:
                for k in conf.ks:
                    start_time = time.time()
                    simp.baseline_k_q(dataset, conf.num_files, k, conf.min_q, conf.q)
                    with open('./stats', 'a') as stats_f:
                        total = time.time() - start_time
                        stats_f.write(f'{conf.dataset} ktruss {k} {conf.min_q} {conf.q} {total}\n')
            else:
                start_time = time.time()
                simp.baseline_k_q(dataset, conf.num_files, conf.k, conf.min_q, conf.q)
                with open('./stats', 'a') as stats_f:
                    total = time.time() - start_time
                    stats_f.write(f'{conf.dataset} ktruss {conf.k} {conf.min_q} {conf.q} {total}\n')
