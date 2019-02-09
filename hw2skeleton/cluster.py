from .utils import Atom, Residue, ActiveSite
import os
import numpy as np
from random import randint
import random
import math
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import matplotlib.pyplot as plt

####################################################
###############similarity functions
####################################################
####################################################

def compute_similarity(site_a, site_b):
#####################overall framework is distilling the data into simpler pieces in stages
#####################in the end the reduced information is used to calculate similarity
    """
    Compute the similarity between two given ActiveSite instances.

    Input: two ActiveSite instances
    Output: the similarity between them (a floating point number)
    """
    similarity = 0.0
    #########simplify data to type of amino acid
    seq_a = calc_seq(site_a)
    seq_b = calc_seq(site_b)
    #########calculate alignment score for simplified alignment
    alignments = pairwise2.align.globalms(seq_a, seq_b, 2, -1, -.5, -.1)
    if alignments == []:
        best_score = 0
    else:
        best_score = alignments[0][2]
    max_score = min([len(seq_a),len(seq_b)])*2
    score1 = float(best_score)/float(max_score)
    #########calculate alignment score for straight aligment
    seq_a = simplify_seq(site_a)
    seq_b = simplify_seq(site_b)
    #########calculate alignment score for simplified alignment
    alignments = pairwise2.align.localxx(seq_a, seq_b)
    if alignments == []:
        best_score = 0
    else:
        best_score = alignments[0][2]
    max_score = min([len(seq_a),len(seq_b)])*2
    score2 = float(best_score)/float(max_score)
    #########weight
    percent_score = 0.5*(score1+score2)
    ####check to make sure score is within range
    if percent_score < 0:
        percent_score = 0
    if percent_score > 1:
        percent_score = 1
    return percent_score

def calc_class(residue):
    #####classify residue by type of amino acid class it falls into: charged, polar, amphipathic, hydrophobic
    #####returns a string
    classdict = {
        'ARG': 'c', 
         'LYS': 'c', 
         'ASP': 'c', 
         'GLU': 'c', 
         'GLN': 'p',
         'ASN': 'p', 
         'HIS': 'p', 
         'SER': 'p', 
         'THR': 'p', 
         'TYR': 'p', 
         'CYS': 'p', 
         'TRP': 'a', 
         'TYR': 'a', 
         'MET': 'a', 
         'ALA': 'h', 
         'ILE': 'h', 
         'LEU': 'h', 
         'PHE': 'h', 
         'VAL': 'h',
         'PRO': 'h',
         'GLY': 'h'}
    ######check if labelled correctly
    if residue.type in classdict:
        ######if so return the class 
        return classdict.get(residue.type)
    else:
        return 'NA'

def simplify_aa(residue):
    #####returns a string
    classdict = {
        'ARG': 'R', 
         'LYS': 'K', 
         'ASP': 'D', 
         'GLU': 'E', 
         'GLN': 'Q',
         'ASN': 'N', 
         'HIS': 'H', 
         'SER': 'S', 
         'THR': 'T', 
         'TYR': 'Y', 
         'CYS': 'C', 
         'TRP': 'W', 
         'TYR': 'Y', 
         'MET': 'M', 
         'ALA': 'A', 
         'ILE': 'I', 
         'LEU': 'L', 
         'PHE': 'F', 
         'VAL': 'V',
         'PRO': 'P',
         'GLY': 'G'}
    ######check if labelled correctly
    if residue.type in classdict:
        ######if so return the class 
        return classdict.get(residue.type)
    else:
        return 'NA'
    
def simplify_seq(activesite):
    final_string = ''
    for residue in activesite.residues:
        final_string += (simplify_aa(residue))
    return(final_string)
    
def calc_seq(activesite):
    final_string = ''
    for residue in activesite.residues:
        final_string += (calc_class(residue))
    return(final_string)

####################################################
###############partitioning clustering supplemental functions
####################################################
####################################################

def find_closest(centroid_list,inputs,DistMatrix):
    closest_vector = []
    #####calculate distances between each item and centroids
    for item in inputs:
        item_index = inputs.index(item)
        sim_vector = []
        ###loop through centroids
        for centroid in centroid_list:
            centroid_index = inputs.index(centroid)
            ###calculate similarity and append to similarity vector for that item
            ###sim_vector.append(compute_similarity(centroid,item))
            sim_vector.append(DistMatrix[item_index][centroid_index])
        #####find most similar centroid for that point
        closest_vector.append(sim_vector.index(max(sim_vector)))
    ######returns a vector same size as table indicating which centroid it is closest to
    
    return(closest_vector)

def find_newcentroid(closest_vector,test,n_clusters,DistMatrix):
    ####split up test vector by cluster identity
    clusters = []
    for i in range(n_clusters):
        clusters.append([])
    i = 0
    for centroid in closest_vector:
        tmp = test[i]
        clusters[centroid].append(tmp)
        i+=1
    #####okay so now that they are split up by cluster find the point with smallest intracluster distance
    new_centroids = []
    for k in clusters:
        dists = []
        for i in range(len(k)):
            ###calc intracluster distance
            ##row contains intracluster distances
            row = []
            for j in range(len(k)):
                sim = DistMatrix[i][j]
                row.append(sim)
            row_sum = sum(row)
            ####this is sum of final distances
            dists.append(row_sum)
        #####find point with smallest distance; this is the new centroid
        if dists != []:
            new_cent = dists.index(min(dists))
            new_centroids.append(k[new_cent])
        else:
            new_cent = random.sample(test, 1)
            print('new random centroid!:',new_cent,new_centroids)
        #####
    return(new_centroids)

####################################################
###############hierarchical clustering supplemental functions
####################################################
####################################################

def calc_distmatrix(active_sites):
    DistMatrix = []
    for item1 in active_sites:
        row = []
        for item2 in active_sites:
            dist = compute_similarity(item1,item2)
            row.append(dist)
        DistMatrix.append(row)
    return DistMatrix

####################################################
############################################Main Clustering Functions
####################################################
####################################################

def cluster_by_partitioning(active_sites,n_clusters):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """
    ######return this
    clustering = []
    ###Randomize cluster centroids
    random.seed(2)
    rnd_nums = random.sample(range(0, len(active_sites)-1), n_clusters)
    centroids = []
    for num in rnd_nums:
        centroids.append(active_sites[num])
    ###Main
    i = 0
    ######calculate distance matrix
    DistMatrix = calc_distmatrix(active_sites)
    print('calcd distances')
    while i < 50:
        ###return vector indicating labels
        labels = find_closest(centroids,active_sites,DistMatrix)
        print('found closests')
        ###Find New Centroids
        new_centroids = find_newcentroid(labels,active_sites,n_clusters,DistMatrix)
        print('found new centroids')
        ###check if final centers have been reached
        if np.all(centroids == new_centroids):
            break
        ###update centroids
        centroids = new_centroids
        i+=1
        print('this is i:',i)
    ####format and return clusters
    for cluster in set(labels):
        tmp_cluster =[]
        ###scroll through active sites and assign it to the apropriate cluster
        for i in range(len(labels)):
            if labels[i] == cluster:
                tmp_cluster.append(active_sites[i])
        clustering.append(tmp_cluster)
    ####return
    return(clustering)

def cluster_hierarchically(active_sites):
    """
    Cluster the given set of ActiveSite instances using an agglomerative hierarchical algorithm.                                                          #

    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """
    ######return this
    clusterings = []
    ######this is the tree which is stored as labels only; used internally 
    clust_all = []
    ######initialize clust, each point is its own cluster to start
    clust = []
    i = 0
    for item in active_sites:
        clust.append(i)
        i+=1
    ######calculate distance matrix
    DistMatrix = calc_distmatrix(active_sites)
    ######Begin agglomeration
    while True:
        closest_clust = []
        for clust_id in clust:
            #####define self and other because we want to find closest neighbor not in same cluster
            self_clust = []
            other_clust = []
            #####iterate through test to assign self and other clusters; appending index for easy access
            for i in range(len(active_sites)):
                if clust[i] == clust_id:
                    self_clust.append(i)
                else:
                    other_clust.append(i)
            #####now use distance matrix to find those points closest to each cluster (find closest neighbors)
            ##min_val holds current minimum, ind_self = index of that item in self, ind_other = index of that item in other
            min_val = -1 ##similarity is 0<sim<1
            ind_self = math.inf
            ind_other = math.inf
            for row in range(len(DistMatrix)):
                ####compare distances between those in and those out of self cluster
                if row in self_clust:
                    for col in range(len(DistMatrix)):
                        if col not in self_clust:
                            ###compare max sim
                            if DistMatrix[row][col] > min_val:
                                ##update
                                min_val = DistMatrix[row][col]
                                ind_self = row
                                ind_other = col
            ####remember which cluster was the closest; store the cluster ID!!
            closest_clust.append(clust[ind_other])
        #####go through clusters and combine clusters based on closest neighbors
        ####neighbors have to be reciprocal; this is written out long form for my own clarity
        for i in range(len(closest_clust)):
            self_index = i
            neighbor_index = closest_clust[i] ##index of neighbor is stored here
            self_neighbor = closest_clust[i] ##index of neighbor is stored here
            neighbor_neighbor = closest_clust[closest_clust[i]] 
            if self_index == neighbor_neighbor:
                ###update clusters
                self_cluster_id = clust[self_index]
                neighbor_cluster_id = clust[neighbor_index]
                ##find neighbor clusters and update to self cluster id
                for j in range(len(clust)):
                    if (clust[j] == neighbor_cluster_id):
                        clust[j] = self_cluster_id
        #####save cluster membership and number of clusters
        clust_all.append(clust.copy())
        #####see if num sets == 1 yet
        if len(set(clust)) == 2:
            break
    #####for one cluster, just avoid running operations and assign everything to 0
    one_clust = [0] * len(active_sites)
    clust_all.append(one_clust)
    ####format and return clusters
    for labels in clust_all:
        n_cluster = []
        for cluster in set(labels):
            tmp_cluster =[]
            ###scroll through active sites and assign it to the apropriate cluster
            for i in range(len(labels)):
                if labels[i] == cluster:
                    tmp_cluster.append(active_sites[i])
            n_cluster.append(tmp_cluster)
        clusterings.append(n_cluster)
    ####return
    return(clusterings)

def calc_clust_dist(a_clustering):
    clust_dist = []
    for clust in a_clustering:
        dists = calc_distmatrix(clust)
        total_dist = 0.0
        for i in range(len(clust)):
            for j in range(len(clust)):
                if j > i:
                    total_dist += dists[i][j]
        #####normalize by number of points in cluster
        score = total_dist/(len(clust)**2)
        clust_dist.append(score)
    return(clust_dist)

def compare_clusterings(part_quality_scores,hier_quality_scores):
    ###########given clusterings from two different algorithms
    ##########return an array containing a list of lists; list[0] = all partitioning cluster quality scores, list[1] = all hierarchical cluster quality scores
    ##########also return a nice violin plot of these values
    clusts_all = []
    ###
    part_all = []
    for clust in part_quality_scores:
        for score in clust:
            part_all.append(score)
    clusts_all.append(part_all)
    ###
    hier_all = []
    for clust in hier_quality_scores:
        for score in clust:
            hier_all.append(score)
    clusts_all.append(hier_all)
    #######plot violinplots
    pos   = [1,2]
    label = ['partitioning','hierarchical']
    data = clusts_all

    plt.figure()
    ax = plt.subplot(111)
    plt.violinplot(data, pos, vert=False)
    ax.set_yticks(pos)
    ax.set_yticklabels(label)
    ###
    return clusts_all

def test_format_part(clustering):
    output = []
    for clust in clustering:
        out_clust = []
        for site in clust:
            out_clust.append(int(site.name))
        output.append(out_clust)
    return(output)

def test_format_hier(clusterings):
    output = []
    for clustering in clusterings:
        out_clustering = []
        for clust in clustering:
            out_clust = []
            for site in clust:
                out_clust.append(int(site.name))
            out_clustering.append(out_clust)
        output.append(out_clustering)
    return(output)
