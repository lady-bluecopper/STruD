# STruD
Algorithms to perform the truss decomposition of simplicial complexes.

## Files Included

| File Name                         | Content                                       |
| ------                            | ------                                        |
| configuration                     | file to set parameters and task to run        |
| external_merge_sort               | procedures for external sort of large files   |
| graph_truss_decomposition         | procedures for standard truss decomposition   |
| run                               | script to launch to run STruD                  |
| simplicial_truss_decomposition    | procedures for simplicial truss decomposition |
| utils                             | utility functions                             |

## Simplicial Truss Decomposition

Open the file *configuration* to set the parameters and the task to perform among:
    - imp: computes and stores the non-trivial simplicial trussness values
    - exp: computes and stores the simplicial trussness of all the simplices in the complex
    - topn: finds the top-n simplices with maximum trussness 
    - ktruss: finds the k-truss of simplices of size greater or equal to q

Then, run the script *run* to start the computation. A log file *stats.log* will be created to log the time required to perform each step of the computation, while a file *stats* will be created to store the total running time of the task.

## Standard Truss Decomposition

To perform the graph decomposition of the 1-skeleton of a simplicial complex, you need to set the name of the input file in the *configuration* file and then run the script *graph_truss_decomposition*. This script generates and stores the 1-skeleton, runs the Networkx truss decomposition algorithm, and stores the results.

## Dataset Format

The simplicial complex must be a list of ordered tuples stored in a pickle file.
The algorithm does not assume that all the simplices in the complex are maximal.

## System Requirements

    networkx=2.4
    python=3.6
