# files and folders
dataset = "dblp_filtered"
data_dir = "../data/"
output_dir = "../output/"

# technical parameters
num_files = 10
max_ram = 30359720775

n = 70  # num of simplices with max trussness to extract
k = 40  # k-truss to compute
q = 20  # max size of simplices to explore
min_q = 2  # min size of simplices to explore

# flags to execute multiple tests 
# (0:imp, 1:exp, 2:topn, 3:ktruss)
experiments = [1,0,0,0]

vary_q = False  # test multiple max sizes q
qs = [2,3]

vary_k = True  # test multiple k truss values
ks = [90,80,70,60,50,40]

vary_n = True # test multiple n values in the top-n task
ns = [50,30,10]