from hw2skeleton import cluster
from hw2skeleton import io
import os

def test_similarity():
    filename_a = os.path.join("data", "276.pdb")
    filename_b = os.path.join("data", "4629.pdb")

    activesite_a = io.read_active_site(filename_a)
    activesite_b = io.read_active_site(filename_b)

    # update this assertion
    assert cluster.compute_similarity(activesite_a, activesite_b) == 0.47000000000000003

def test_partition_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))
    ####format answer
    #clust1 = [276]
    #clust2 = [4629, 10701]
    ##
    #answer = []
    #tmp1 = []
    #for id in clust1:
   #     filepath = os.path.join("data", "%i.pdb"%id)
   #     tmp1.append(io.read_active_site(filepath))
   # tmp2 = []
   # for id in clust2:
  #      filepath = os.path.join("data", "%i.pdb"%id)
   #     tmp2.append(io.read_active_site(filepath))
   # answer.append(tmp1)
   # answer.append(tmp2)
    # update this assertion
    assert test_format(cluster.cluster_by_partitioning(active_sites,2)) == [[276], [4629, 10701]]


def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    # update this assertion
    assert cluster.cluster_hierarchically(active_sites) == [[[276, 4629], [10701]], [[276, 4629, 10701]]]

