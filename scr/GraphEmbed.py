from multiprocessing import cpu_count, Pool, get_logger, log_to_stderr
from param_parser import parameter_parser
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
import gensim
import pkg_resources
import json
import traceback
import numpy as np
import torch.nn as nn
import torch
from utils import *
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, fbeta_score

args = parameter_parser()

def error(msg, *args):
    return get_logger().error(msg, *args)

class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable
        return

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result
    
class Node2Vec():

    def __init__(self, G, weights=None, p=1, q=1, n_walks=100, dimensions=12, walk_length=5, workers: int = 1) -> None:
        super(Node2Vec, self).__init__()
        self.p = p
        self.q = q
        self.n_walks = n_walks
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.graph = G
        self.weights = weights
        self.workers = workers

    def _run(self):
        self._precompute_probabilities(self.graph)
        self._generate_walks()


    def _precompute_probabilities(self, graph:nx.Graph=None):
        '''
        计算转移概率
        '''
        if graph is None:
            graph = self.graph
        nodes_generator = graph.nodes()
        self.distance_dict = defaultdict(dict)     # 直接相连的节点会被记录
        for source in tqdm(nodes_generator, desc='Computing transition probabilities'):
            # 计算每个节点到邻居节点的转移概率
            nbrs = nx.neighbors(graph, source)  # 起始节点的邻居
            for cur_node in nbrs:
                try:
                    min_dis = self.distance_dict[source][cur_node]
                    dis = 1
                    if dis < min_dis:
                        self.distance_dict[source][cur_node] = dis    # 两节点直接相连，距离为1
                        self.distance_dict[cur_node][source] = dis
                except:
                    self.distance_dict[source][cur_node] = 1    # 两节点直接相连，距离为1
                    self.distance_dict[cur_node][source] = 1
    
    def _generate_walks(self) -> list:
        '''
        随机游走
        '''
        num_cores = max(cpu_count() - 6, 1)
        print(f'num of cpu cores:{cpu_count()}; using {num_cores} cores')
        nodes_generator = list(self.graph.nodes())
        sources_cores = []
        
        starts = []
        for c in range(0, len(nodes_generator), int(len(nodes_generator)/num_cores)):
            starts.append(c)
        starts.pop(0)
        starts.append(len(nodes_generator))
        start = 0
        for c in starts:
            end = c
            sources = nodes_generator[start: end]
            sources_cores.append(sources)
            start = end

        self.walks = []
        log_to_stderr()
        pool = Pool(num_cores)
        for i in range(len(sources_cores)):
            sources = sources_cores[i]
            pool.apply_async(func=LogExceptions(self._sub_generate_walks), args=(sources,i), callback=self.walks.extend)   # get方法取返回的值

        pool.close()    # 进程池不在接受任务
        pool.join()     # 进程池阻塞主进程
        # self.walks.extend(self._sub_generate_walks( sources_cores[0], 0))

        return self.walks



    def _sub_generate_walks(self, sources, core_id):
        
        walks = []
        for source in tqdm(sources, desc=f'generate walks_{core_id}'):
            for l in range(self.n_walks):
                nbrs = nx.neighbors(self.graph, source)
                nbrs = list(nbrs)
                degree = len(nbrs)
                if degree == 0:
                    walk = [source] * self.walk_length
                    walks.append(walk)
                    break
                idx = np.random.choice(degree, 1)[0]  # 第一次随机游走到下一节点
                v = nbrs[idx]
                t = source      # 初始节点变为上一节点
                k = 0
                walk = [v]
                while k < self.walk_length:
                    nbrs = nx.neighbors(self.graph, v)
                    nbrs = list(nbrs)
                    degree = len(nbrs)

                    # 为下一节点分配概率，需要结合上一节点t计算概率
                    pi = []
                    weights = []
                    for node in nbrs:
                        if node == t:
                            pi.append(1 / self.p)
                        elif self.distance_dict[node].get(t, -1) == -1:
                            # 两节点距离大于2
                            pi.append(1 / self.q)
                        else:
                            pi.append(1)   # 两节点直接相连
                        
                        if self.weights is None:
                            weights.append(1)
                        else:
                            weights.append(self.weights[int(v),int(node)])

                    # 加上权重归一化后得到概率
                    weights = np.array(weights)
                    pi = np.array(pi)
                    prob = np.ones_like(pi) * weights * pi
                    
                    # if degree == 0:
                    #     print(f'degree:{degree}')
                    # prob = (prob / degree) * pi * weights
                    prob = prob / np.sum(prob)
                    # res = 1 - np.sum(prob)
                    # prob[-1] = prob[-1] + res
                    
                    # 选择下一个邻居
                    idx = np.random.choice(degree, 1, p=prob)[0]
                    t = v
                    v = nbrs[idx]
                    walk.append(int(v))
                    k += 1
                
                walks.append(walk)
        
        return walks


    def fit(self, **skip_gram_params) -> gensim.models.Word2Vec:
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameters for gensim.models.Word2Vec - do not supply 'size' / 'vector_size' it is
            taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        # Figure out gensim version, naming of output dimensions changed from size to vector_size in v4.0.0
        gensim_version = pkg_resources.get_distribution("gensim").version
        size = 'size' if gensim_version < '4.0.0' else 'vector_size'
        if size not in skip_gram_params:
            skip_gram_params[size] = self.dimensions

        if 'sg' not in skip_gram_params:
            skip_gram_params['sg'] = 1

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)