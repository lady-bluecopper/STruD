#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
from itertools import combinations
from external_merge_sort import externamMergeSort
import numpy as np
import utils as ut
import configuration as conf
import pickle
import time
import os
import logging
import sys
logging.basicConfig(filename='stats.log',level=logging.INFO)


# finds connected components of simplicial complex
def connected_components(K):
    CC = []
    CC_v = []
    cands = sorted(K, key=lambda x: -len(x))
    CC.append([cands[0]])
    CC_v.append(set(cands[0]))
    stop = False
    for s in cands[1:]:
        for v in s:
            for idx in range(len(CC_v)):
                if v in CC_v[idx]:
                    CC_v[idx].update(s)
                    CC[idx].append(s)
                    stop = True
                    break
            if stop:
                break
        if stop:
            stop = False
        else:
            CC.append([s])
            CC_v.append(set(s))
    return CC


# adds neighbors to dictionary nbs
def merge(nbs, n_s, idx, C):
    for y in n_s:
        tmp = set(C[y])
        for v in C[idx]:
            l = len(tmp)
            tmp.discard(v)
            if len(tmp) == l:
                first_v = v
        second_v = tmp.pop()
        if second_v > C[idx][-1]:
            nbs[idx].add(y)
        elif first_v > C[y][-1]:
            nbs[y].add(idx)
    return nbs


# extends simplices in S by adding a new vertex
def extend_simplices(S, K, q):
    C = set()
    if q == 2 or len(S) == 0:
        for s in K:
            if len(s) == q:
                C.add(tuple(sorted(s)))
            elif len(s) > q:
                for c in combinations(s, q):
                    C.add(tuple(sorted(c)))
        return list(C)
    inv_index = defaultdict(set)
    for idx in range(len(K)):
        s = K[idx]
        if len(s) >= q:
            for v in s:
                inv_index[v].add(idx)
    if len(inv_index) == 0:
        return []
    for s in S:
        tmp = inv_index[s[0]]
        for idx in range(1, len(s)):
            tmp = tmp & inv_index[s[idx]]
        if len(tmp) > 0:
            tmp = set(ut.flatmap(lambda x: K[x], tmp))
            for v in s:
                tmp.remove(v)
            for e in tmp:
                ext = list(s)
                ext.append(e)
                C.add(tuple(sorted(ext)))
    return list(C)


# finds joists
def initialize_neighbours(C, num_files, max_ram):
    start = time.time()
    nbs = defaultdict(set)
    I = defaultdict(list)
    in_memory = True
    pointers = []
    for i in range(num_files):
        file_name = f'tmp_{i}'
        tmp = open(conf.data_dir + file_name, "wb")
        pointers.append(tmp)
    for idx in range(len(C)):
        s = C[idx]
        curr = find_cocofaces(idx, I, C)
        nbs = merge(nbs, curr, idx, C)
        w = len(s) - 1
        step = 1
        I[s[:w]].append(idx)
        while step < w + 1:
            code = list(s[:step - 1])
            code.extend(s[step:step + w])
            I[tuple(code)].append(idx)
            step += 1
        if idx % 10000 == 0:
            size_1 = sum(sys.getsizeof(x) for x in nbs.values()) + sum(sys.getsizeof(x) for x in nbs.keys())
            size_2 = sum(sys.getsizeof(x) for x in I.values()) + sum(sys.getsizeof(x) for x in I.keys())
            if size_1 + size_2 > max_ram:
                in_memory = False
                for e in nbs:
                    for n in nbs[e]:
                        pointers[e % num_files].write((f'{e} {n}\n').encode(encoding='UTF-8'))
                nbs.clear()
    if not in_memory:
        for e in nbs:
            for n in nbs[e]:
                pointers[e % num_files].write((f'{e} {n}\n').encode(encoding='UTF-8'))
        nbs.clear()
        for i in range(num_files):
            pointers[i].close()
    end = time.time()
    logging.info(f'candidates {end - start}')
    start = time.time()
    if not in_memory:
        tmp_size = os.stat(conf.data_dir + "tmp_0").st_size
        if tmp_size > max_ram:
            external_sort(num_files)
            nbs = validate_from_large_files(C, num_files)
        else:
            nbs = validate_from_files(C, num_files)
    else:
        nbs = validate_neighbours(C, nbs)
    logging.info(f'validation {time.time() - start}')
    return nbs


# validates joists
def validate_neighbours(C, nbs):
    inv_idx = defaultdict(set)
    for idx in range(len(C)):
        for v in C[idx]:
            inv_idx[v].add(idx)
    real_nbs = defaultdict(set)
    for s, nb_set in nbs.items():
        v_set = set(ut.flatmap(lambda x: C[x], nb_set))
        for v in C[s]:
            v_set.discard(v)
        for v in v_set:
            if v > C[s][-1]:
                cand_joist = inv_idx[v] & nb_set
                if len(cand_joist) == len(C[s]):
                    real_nbs[C[s]].add(v)
                    for c in cand_joist:
                        w = C[s][0]
                        nxt = 1
                        while w in C[c]:
                            w = C[s][nxt]
                            nxt += 1
                        real_nbs[C[c]].add(w)
    return real_nbs


# finds simplices that share q-1 vertices with s
def find_cocofaces(idx, I, C):
    n_s = set()
    s = C[idx]
    w = len(s) - 1
    step = 1
    for y in I[s[:w]]:
        n_s.add(y)
    while step < w + 1:
        code = list(s[:step - 1])
        code.extend(s[step:step + w])
        for y in I[tuple(code)]:
            n_s.add(y)
        step += 1
    return n_s


# finds joists - external
def initialize_neighbours_external(C, num_files, max_size):
    start = time.time()
    nbs = defaultdict(set)
    I = defaultdict(list)
    lines = 0
    max_real_num = 0
    pointers = []
    for i in range(num_files):
        file_name = f'tmp_{i}'
        tmp = open(conf.data_dir + file_name, "wb")
        pointers.append(tmp)
    for idx in range(len(C)):
        s = C[idx]
        curr = find_cocofaces(idx, I, C)
        nbs = merge(nbs, curr, idx)
        lines += 2 * len(curr)
        if lines > 1000000:
            for e in nbs:
                for n in nbs[e]:
                    pointers[e % num_files].write((f'{e} {n}\n').encode(encoding='UTF-8'))
            nbs.clear()
            lines = 0
        w = len(s) - 1
        step = 1
        I[s[:w]].append(idx)
        while step < w + 1:
            code = list(s[:step - 1])
            code.extend(s[step:step + w])
            I[tuple(code)].append(idx)
            step += 1
        if lines > 0:
            for e in nbs:
                for n in nbs[e]:
                    pointers[e % num_files].write((f'{e} {n}\n').encode(encoding='UTF-8'))
            nbs.clear()
            lines = 0
    for i in range(num_files):
        pointers[i].close()
    logging.info(f'candidates {time.time() - start}')
    start = time.time()
    tmp_size = os.stat(conf.data_dir + "tmp_0").st_size
    if tmp_size > max_size:
        external_sort(num_files)
        nbs = validate_from_large_files(C, num_files)
    else:
        nbs = validate_from_files(C, num_files)
    logging.info(f'validation {time.time() - start}')
    return nbs


# sorts large files
def external_sort(num_files):
    for n in range(num_files):
        try:
            f = f'{conf.data_dir}tmp_{n}'
            obj = externamMergeSort()
            # max lines per file
            obj.splitFiles(f, 100000000)
            obj.mergeSortedtempFiles_low_level(f + '_sorted')
            os.remove(f)
        except FileNotFoundError:
            print(f'{f} not found!')
    return 'Done'


# improved for finding and validating each joist just once
def validate_from_files(C, num_files):
    inv_idx = defaultdict(set)
    for idx in range(len(C)):
        for v in C[idx]:
            inv_idx[v].add(idx)
    real_nbs = defaultdict(set)
    for i in range(num_files):
        f = f'{conf.data_dir}tmp_{i}'
        start = time.time()
        nbs = defaultdict(set)
        v_sets = defaultdict(set)
        with open(f, 'rb') as in_f:
            for line in in_f.readlines():
                if line:
                    lst = line.decode().strip().split(" ")
                    s = int(lst[0])
                    n = int(lst[1])
                    nbs[s].add(n)
                    v_sets[s].update(C[n])
        for s in v_sets.keys():
            for v in C[s]:
                v_sets[s].discard(v)
            for v in v_sets[s]:
                if v > C[s][-1]:
                    cand_joist = inv_idx[v] & nbs[s]
                    if len(cand_joist) == len(C[s]):
                        real_nbs[C[s]].add(v)
                        for c in cand_joist:
                            w = C[s][0]
                            nxt = 1
                            while w in C[c]:
                                w = C[s][nxt]
                                nxt += 1
                            real_nbs[C[c]].add(w)
        print(f'processed {time.time() - start}')
    return real_nbs


# improved for finding each joist just once - external
def validate_from_large_files(C, num_files):
    inv_idx = defaultdict(set)
    for idx in range(len(C)):
        for v in C[idx]:
            inv_idx[v].add(idx)
    real_nbs = defaultdict(set)
    for i in range(num_files):
        f = f'{conf.data_dir}tmp_{i}_sorted'
        start = time.time()
        with open(f, 'rb') as in_f:
            lst = in_f.readline().decode().strip().split(" ")
            s = int(lst[0])
            nb_set = set()
            v_set = set()
            end = False
            while lst:
                while int(lst[0]) == s:
                    nb_set.add(int(lst[1]))
                    v_set.update(C[int(lst[1])])
                    line = in_f.readline().decode().strip()
                    if line:
                        lst = line.split(" ")
                    else:
                        end = True
                        break
                for v in C[s]:
                    v_set.discard(v)
                for v in v_set:
                    if v > C[s][-1]:
                        cand_joist = inv_idx[v] & nb_set
                        if len(cand_joist) == len(C[s]):
                            real_nbs[C[s]].add(v)
                            for c in cand_joist:
                                w = C[s][0]
                                nxt = 1
                                while w in C[c]:
                                    w = C[s][nxt]
                                    nxt += 1
                                real_nbs[C[c]].add(w)
                if not end:
                    s = int(lst[0])
                    nb_set = set()
                    v_set = set()
                else:
                    break
        print(f'processed {time.time() - start}')
    return real_nbs


# finds and stores all the trussness values
def baseline_expl(K, num_files, stop):
    total_start = time.time()
    start_time = time.time()
    CC = connected_components(K)
    out_files = []
    logging.info(f'CC {time.time()-start_time}')
    for c in CC:
        d = min(max(len(x) for x in c), stop)
        q = 2
        S = []
        J = defaultdict(set)
        for idx in range(len(c)):
            for v in c[idx]:
                J[v].add(idx)
        while q <= d:
            if q > 2 and len(S) == 0:
                break
            start_time = time.time()
            C = extend_simplices(S, c, q)
            logging.info(f'{len(C)} {q} extend {time.time()-start_time}')
            if len(C) == 0:
                break
            lb = defaultdict(int)
            for idx in range(len(C)):
                s = C[idx]
                tmp = J[s[0]]
                for j in range(1, len(s)):
                    tmp = tmp & J[s[j]]
                if len(tmp) > 0:
                    lb[s] = max(map(lambda x: len(c[x]) - len(s), tmp))
            nbs = initialize_neighbours(C, num_files, conf.max_ram)
            tr = {s: len(nbs[s]) for s in nbs}
            S = []
            queue = defaultdict(set)
            for v, trussness in tr.items():
                queue[trussness].add(v)
            start_time = time.time()
            while len(queue) > 0:
                curr_k = min(queue.keys())
                curr_vs = list(queue[curr_k])
                mod = False
                for s in curr_vs:
                    if mod:
                        break
                    for comb in combinations(s, q - 1):
                        for v in nbs[s]:
                            tmp = list(comb)
                            tmp.append(v)
                            n = tuple(sorted(tmp))
                            if n in tr and tr[n] > tr[s]:
                                to_remove = s[0]
                                nxt = 1
                                while to_remove in comb:
                                    to_remove = s[nxt]
                                    nxt += 1
                                nbs[n].discard(to_remove)
                                new_tr = len(nbs[n])
                                if new_tr != tr[n]:
                                    queue[tr[n]].remove(n)
                                    if len(queue[tr[n]]) == 0:
                                        queue.pop(tr[n])
                                    if new_tr > 0:
                                        tr[n] = new_tr
                                        queue[tr[n]].add(n)
                                    else:
                                        del tr[n]
                                if new_tr < curr_k:
                                    mod = True
                    S.append(s)
                    queue[curr_k].remove(s)
                    if len(queue[curr_k]) == 0:
                        queue.pop(curr_k)
            logging.info(f'trussness {time.time()-start_time}')
            if len(out_files) < q - 1:
                out_files.append(open(f'{conf.output_dir}/{conf.dataset}_tr_exp_q={q}.pkl', 'wb'))
            ut.append_obj(tr, out_files[q - 2])
            q += 1
    for f in out_files:
        f.close()
    logging.info(f'total {time.time() - total_start}')


# finds and stores only the non-trivial trussness values
def baseline_impl(K, num_files, stop):
    start_time = time.time()
    total_start = time.time()
    CC = connected_components(K)
    out_files = []
    logging.info(f'CC {time.time()-start_time}')
    for c in CC:
        d = min(max(len(x) for x in c), stop)
        q = 2
        J = defaultdict(set)
        S = []
        for idx in range(len(c)):
            for v in c[idx]:
                J[v].add(idx)
        while q <= d:
            if q > 2 and len(S) == 0:
                break
            start_time = time.time()
            C = extend_simplices(S, c, q)
            if len(C) == 0:
                break
            logging.info(f'{len(C)} {q} extend {time.time()-start_time}')
            lb = defaultdict(int)
            for idx in range(len(C)):
                s = C[idx]
                tmp = J[s[0]]
                for j in range(1, len(s)):
                    tmp = tmp & J[s[j]]
                if len(tmp) > 0:
                    lb[s] = max(map(lambda x: len(c[x]) - len(s), tmp))
            nbs = initialize_neighbours(C, num_files, conf.max_ram)
            tr = {s: len(nbs[s]) for s in nbs}
            to_examine = True
            for s in tr.keys():
                if lb[s] != tr[s]:
                    to_examine = False
                    break
            if to_examine:
                for s in tr.keys():
                    if tr[s] > 0:
                        S.append(s)
                q += 1
                continue
            S = []
            queue = defaultdict(set)
            for v, trussness in tr.items():
                queue[trussness].add(v)
            start_time = time.time()
            while len(queue) > 0:
                curr_k = min(queue.keys())
                curr_vs = list(queue[curr_k])
                mod = False
                for s in curr_vs:
                    if mod:
                        break
                    for comb in combinations(s, q - 1):
                        for v in nbs[s]:
                            tmp = list(comb)
                            tmp.append(v)
                            n = tuple(sorted(tmp))
                            if n in tr and tr[n] > tr[s]:
                                to_remove = s[0]
                                nxt = 1
                                while to_remove in comb:
                                    to_remove = s[nxt]
                                    nxt += 1
                                nbs[n].discard(to_remove)
                                new_tr = len(nbs[n])
                                if new_tr != tr[n]:
                                    queue[tr[n]].remove(n)
                                    if len(queue[tr[n]]) == 0:
                                        queue.pop(tr[n])
                                    if new_tr > 0:
                                        tr[n] = new_tr
                                        queue[tr[n]].add(n)
                                    else:
                                        del tr[n]
                                if new_tr < curr_k:
                                    mod = True
                    S.append(s)
                    queue[curr_k].remove(s)
                    if len(queue[curr_k]) == 0:
                        queue.pop(curr_k)
            logging.info(f'trussness {time.time()-start_time}')
            for s in list(tr.keys()):
                if tr[s] == lb[s]:
                    del tr[s]
            if len(out_files) < q - 1:
                out_files.append(open(f'{conf.output_dir}{conf.dataset}_tr_imp_q={q}.pkl', 'wb'))
            ut.append_obj(tr, out_files[q - 2])
            q += 1
    for f in out_files:
        f.close()
    logging.info(f'total {time.time() - total_start}')


# K-Truss of q-simplices with q >= t
def baseline_k_q(K, num_files, k, t, u):
    start_time = time.time()
    total_start = time.time()
    CC = connected_components(K)
    logging.info(f'CC {time.time()-start_time}')
    out_file = open(f'{conf.output_dir}{conf.dataset}_tr_k={k}_q>{t}<{u}.pkl', 'wb')
    for c in CC:
        S = []
        d = min(max(len(x) for x in c), u)
        q = t
        J = defaultdict(set)
        for idx in range(len(c)):
            for v in c[idx]:
                J[v].add(idx)
        while q <= d:
            if q > t and len(S) == 0:
                break
            start_time = time.time()
            C = extend_simplices(S, c, q)
            if len(C) == 0:
                break
            logging.info(f'{len(C)} {q} extend {time.time()-start_time}')
            lb = defaultdict(int)
            for idx in range(len(C)):
                s = C[idx]
                tmp = J[s[0]]
                for j in range(1, len(s)):
                    tmp = tmp & J[s[j]]
                if len(tmp) > 0:
                    lb[s] = max(map(lambda x: len(c[x]) - len(s), tmp))
            nbs = initialize_neighbours(C, num_files, conf.max_ram)
            tr = {s: len(nbs[s]) for s in nbs}
            to_examine = True
            for s in tr.keys():
                if lb[s] != tr[s]:
                    to_examine = False
                    break
            if to_examine:
                for s in tr.keys():
                    if tr[s] >= k:
                        S.append(s)
                q += 1
                continue
            S = []
            queue = defaultdict(set)
            for v, trussness in tr.items():
                queue[trussness].add(v)
            start_time = time.time()
            while len(queue) > 0:
                curr_k = min(queue.keys())
                curr_vs = list(queue[curr_k])
                mod = False
                for s in curr_vs:
                    if mod:
                        break
                    for comb in combinations(s, q - 1):
                        for v in nbs[s]:
                            tmp = list(comb)
                            tmp.append(v)
                            n = tuple(sorted(tmp))
                            if n in tr and tr[n] > tr[s]:
                                to_remove = s[0]
                                nxt = 1
                                while to_remove in comb:
                                    to_remove = s[nxt]
                                    nxt += 1
                                nbs[n].discard(to_remove)
                                new_tr = len(nbs[n])
                                if new_tr != tr[n]:
                                    queue[tr[n]].remove(n)
                                    if len(queue[tr[n]]) == 0:
                                        queue.pop(tr[n])
                                    if new_tr > 0:
                                        tr[n] = new_tr
                                        queue[tr[n]].add(n)
                                    else:
                                        del tr[n]
                                if new_tr < curr_k:
                                    mod = True
                    if tr[s] >= k:
                        S.append(s)
                    queue[curr_k].remove(s)
                    if len(queue[curr_k]) == 0:
                        queue.pop(curr_k)
            logging.info(f'trussness {time.time()-start_time}')
            for s in list(tr.keys()):
                if tr[s] == lb[s] or tr[s] != k:
                    del tr[s]
            ut.append_obj(tr, out_file)
            q += 1
    out_file.close()
    logging.info(f'total {time.time() - total_start}')


# generates simplices of size q
def extend_simplices_topn(K, q):
    C = set()
    for s in K:
        if len(s) == q:
            C.add(tuple(sorted(s)))
        elif len(s) > q:
            for c in combinations(s, q):
                C.add(tuple(sorted(c)))
    return list(C)


# top-n q-simplices with max trussness
def baseline_top_q(K, num_files, n, q):
    start_time = time.time()
    total_start = time.time()
    CC = connected_components(K)
    logging.info(f'CC {time.time()-start_time}')
    out_file = open(f'{conf.output_dir}{conf.dataset}_tr_top={n}_q={q}.pkl', 'wb')
    past_worst = -1
    for c in CC:
        J = defaultdict(set)
        for idx in range(len(c)):
            for v in c[idx]:
                J[v].add(idx)
        start_time = time.time()
        C = extend_simplices_topn(c, q)
        if len(C) == 0:
            continue
        logging.info(f'{len(C)} {q} extend {time.time()-start_time}')
        lb = defaultdict(int)
        for idx in range(len(C)):
            s = C[idx]
            tmp = J[s[0]]
            for j in range(1, len(s)):
                tmp = tmp & J[s[j]]
            if len(tmp) > 0:
                lb[s] = max(map(lambda x: len(c[x]) - len(s), tmp))
        nbs = initialize_neighbours(C, num_files, conf.max_ram)
        tr = {s: len(nbs[s]) for s in nbs}
        topk = []
        topk_simp = []
        # sort in descending order
        queue = defaultdict(set)
        for v, trussness in tr.items():
            queue[trussness].add(v)
        start_time = time.time()
        finished = False
        while len(queue) > 0:
            curr_k = max(queue.keys())
            curr_vs = list(queue[curr_k])
            if (len(topk) == n and curr_k <= topk[-1]) or curr_k < past_worst:
                break
            for s in curr_vs:
                if len(topk) == n and curr_k <= topk[-1]:
                    break
                for v in list(nbs[s]):
                    if len(topk) == n and tr[s] <= topk[-1]:
                        break
                    for comb in combinations(s, q - 1):
                        tmp = list(comb)
                        tmp.append(v)
                        m = tuple(sorted(tmp))
                        if m in tr and tr[m] < tr[s]:
                            nbs[s].discard(v)
                            new_tr = len(nbs[s])
                            if new_tr > 0:
                                tr[s] = new_tr
                            else:
                                del tr[s]
                            break
                queue[curr_k].remove(s)
                if len(queue[curr_k]) == 0:
                    queue.pop(curr_k)
                if len(topk) == 0 or topk[-1] >= tr[s]:
                    topk.append(tr[s])
                    topk_simp.append(s)
                else:
                    i = 0
                    while topk[i] > tr[s]:
                        i += 1
                    topk.insert(i, tr[s])
                    topk_simp.insert(i, s)
                topk = topk[:n]
                topk_simp = topk_simp[:n]
        if past_worst == -1:
            past_worst = topk[-1]
        elif len(topk) > 0:
            past_worst = max(past_worst, topk[-1])
        logging.info(f'trussness {time.time()-start_time}')
        truss = {topk_simp[i]: topk[i] for i in range(len(topk))}
        ut.append_obj(truss, out_file)
    out_file.close()
    logging.info(f'total {time.time() - total_start}')
