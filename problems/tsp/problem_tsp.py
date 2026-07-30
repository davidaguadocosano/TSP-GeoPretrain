from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search
from utils2.data_utils import rotate_nodes

import subprocess
import tempfile
import time
import re

def nearest_neighbor_graph(nodes, neighbors, knn_strat):
    """Returns k-Nearest Neighbor graph as a **NEGATIVE** adjacency matrix
    """
    num_nodes = len(nodes)
    # If `neighbors` is a percentage, convert to int
    if knn_strat == 'percentage':
        neighbors = int(num_nodes * neighbors)
    
    if neighbors >= num_nodes-1 or neighbors == -1:
        W = np.zeros((num_nodes, num_nodes))
    else:
        # Compute distance matrix
        W_val = squareform(pdist(nodes, metric='euclidean'))
        W = np.ones((num_nodes, num_nodes))
        
        # Determine k-nearest neighbors for each node
        knns = np.argpartition(W_val, kth=neighbors, axis=-1)[:, neighbors::-1]
        # Make connections
        for idx in range(num_nodes):
            W[idx][knns[idx]] = 0
    
    # Remove self-connections
    np.fill_diagonal(W, 1)
    return W


def tour_nodes_to_W(tour_nodes):
    """Computes edge adjacency matrix representation of tour
    """
    num_nodes = len(tour_nodes)
    tour_edges = np.zeros((num_nodes, num_nodes))
    for idx in range(len(tour_nodes) - 1):
        i = tour_nodes[idx]
        j = tour_nodes[idx + 1]
        tour_edges[i][j] = 1
        tour_edges[j][i] = 1
    # Add final connection
    tour_edges[j][tour_nodes[0]] = 1
    tour_edges[tour_nodes[0]][j] = 1
    return tour_edges


class TSP(object):
    """Class representing the Travelling Salesman Problem
    """

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        """Returns TSP tour length for given graph nodes and tour permutations

        Args:
            dataset: graph nodes (torch.Tensor)
            pi: node permutations representing tours (torch.Tensor)

        Returns:
            TSP tour length, None
        """
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour:\n{}\n{}".format(dataset, pi)

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(nodes, graph, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        """Method to call beam search, given TSP samples and a model
        """

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(nodes, graph)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            nodes, graph, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)
    
    
class TSPSL(TSP):
    """Class representing the Travelling Salesman Problem, trained with Supervised Learning
    """

    NAME = 'tspsl'


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, min_size=20, max_size=50, batch_size=128,
                 num_samples=128000, offset=0, distribution=None, neighbors=20, 
                 knn_strat=None, supervised=False, nar=False):
        """Class representing a PyTorch dataset of TSP instances, which is fed to a dataloader

        Args:
            filename: File path to read from (for SL)
            min_size: Minimum TSP size to generate (for RL)
            max_size: Maximum TSP size to generate (for RL)
            batch_size: Batch size for data loading/batching
            num_samples: Total number of samples in dataset
            offset: Offset for loading from file
            distribution: Data distribution for generation (unused)
            neighbors: Number of neighbors for k-NN graph computation
            knn_strat: Strategy for computing k-NN graphs ('percentage'/'standard')
            supervised: Flag to enable supervised learning
            nar: Flag to indicate Non-autoregressive decoding scheme, which uses edge-level groundtruth

        Notes:
            `batch_size` is important to fix across dataset and dataloader,
            as we are dealing with TSP graphs of variable sizes. To enable
            efficient training without DGL/PyG style sparse graph libraries,
            we ensure that each batch contains dense graphs of the same size.
        """
        super(TSPDataset, self).__init__()

        self.filename = filename
        self.min_size = min_size
        self.max_size = max_size
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.offset = offset
        self.distribution = distribution
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.supervised = supervised
        self.nar = nar

        # Loading from file (usually used for Supervised Learning or evaluation)
        if filename is not None:
            self.nodes_coords = []
            self.tour_nodes = []

            print('\nLoading from {}...'.format(filename))
            for line in tqdm(open(filename, "r").readlines()[offset:offset+num_samples], ascii=True):
                line = line.split(" ")
                num_nodes = int(line.index('output')//2)
                self.nodes_coords.append(
                    [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
                )

                if self.supervised:
                    # Convert tour nodes to required format
                    # Don't add final connection for tour/cycle
                    tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
                    self.tour_nodes.append(tour_nodes)

        # Generating random TSP samples (usually used for Reinforcement Learning)
        else:
            # Sample points randomly in [0, 1] square
            self.nodes_coords = []

            print('\nGenerating {} samples of TSP{}-{}...'.format(num_samples, min_size, max_size))
            for _ in tqdm(range(num_samples//batch_size), ascii=True):
                # Each mini-batch contains graphs of the same size
                # Graph size is sampled randomly between min and max size
                num_nodes = np.random.randint(low=min_size, high=max_size+1)
                self.nodes_coords += list(np.random.random([batch_size, num_nodes, 2]))
        
        self.size = len(self.nodes_coords)
        assert self.size % batch_size == 0, \
            "Number of samples ({}) must be divisible by batch size ({})".format(self.size, batch_size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        nodes = self.nodes_coords[idx]
        item = {
            'nodes': torch.FloatTensor(nodes),
            'graph': torch.ByteTensor(nearest_neighbor_graph(nodes, self.neighbors, self.knn_strat))
        }
        if self.supervised:
            # Add groundtruth labels in case of SL
            tour_nodes = self.tour_nodes[idx]
            item['tour_nodes'] = torch.LongTensor(tour_nodes)
            if self.nar:
                # Groundtruth for NAR decoders is the TSP tour in adjacency matrix format
                item['tour_edges'] = torch.LongTensor(tour_nodes_to_W(tour_nodes))

        return item

#dac
class TSPPretrainDataset(TSPDataset):
    def __init__(self, *args, **kwargs):
        # Forzamos que no sea supervisado ni NAR para el pre-entrenamiento
        kwargs['supervised'] = False
        kwargs['nar'] = False
        super(TSPPretrainDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        # Obtenemos los nodos originales (v1)
        nodes_v1 = torch.FloatTensor(self.nodes_coords[idx])
        
        # Generamos un ángulo aleatorio y rotamos para obtener la vista 2 (v2)
        angle = np.random.rand() * 360
        nodes_v2 = rotate_nodes(nodes_v1, angle)

        # Importante: Ambos necesitan su matriz de adyacencia (grafo k-NN)
        # Reutilizamos la función del proyecto original
        graph_v1 = torch.ByteTensor(nearest_neighbor_graph(nodes_v1, self.neighbors, self.knn_strat))
        graph_v2 = torch.ByteTensor(nearest_neighbor_graph(nodes_v2, self.neighbors, self.knn_strat))

        return {
            'nodes_v1': nodes_v1,
            'graph_v1': graph_v1,
            'nodes_v2': nodes_v2,
            'graph_v2': graph_v2
        }

#dac
def solve_concorde(nodes, executable="concorde"):
    """Llama al binario de Concorde para resolver el TSP de forma exacta."""
    num_nodes = len(nodes)
    # Concorde suele trabajar mejor con enteros, escalamos las coordenadas [0,1]
    scale = 1000000
    
    with tempfile.TemporaryDirectory() as tmpdir:
        node_file = os.path.join(tmpdir, "problem.tsp")
        with open(node_file, "w") as f:
            f.write(f"NAME : problem\nTYPE : TSP\nDIMENSION : {num_nodes}\n")
            f.write("EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
            for i, (x, y) in enumerate(nodes):
                f.write(f"{i+1} {int(x*scale)} {int(y*scale)}\n")
            f.write("EOF\n")

        # Ejecutar Concorde
        sol_file = "problem.sol" # Concorde genera este archivo por defecto
        cmd = [executable, "-o", sol_file, node_file]
        
        # Redirigimos salida para que no ensucie la terminal
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, cwd=tmpdir)
        
        # Leer la solución
        with open(os.path.join(tmpdir, sol_file), "r") as f:
            lines = f.readlines()
            # El primer número es la dimensión, el resto es el tour
            tour = [int(x) for x in " ".join(lines).split()][1:]
            
        # Calcular el coste real (usando coordenadas originales)
        # Reutilizamos la lógica de coste del proyecto o calculamos L2
        nodes_torch = torch.FloatTensor(nodes).unsqueeze(0)
        tour_torch = torch.LongTensor(tour).unsqueeze(0)
        from problems.tsp.problem_tsp import TSP
        cost, _ = TSP.get_costs(nodes_torch, tour_torch)
        
        return cost.item(), tour

def solve_lkh(nodes, executable="/home/pfc/dac/learning-tsp/LKH-3.0.9/LKH"):
    """Llama al binario de LKH-3 para resolver el TSP."""
    num_nodes = len(nodes)
    scale = 1000000
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tsp_file = os.path.join(tmpdir, "problem.tsp")
        out_file = os.path.join(tmpdir, "problem.out")
        par_file = os.path.join(tmpdir, "problem.par")
        
        # 1. Escribir archivo .tsp
        with open(tsp_file, "w") as f:
            f.write(f"NAME : problem\nTYPE : TSP\nDIMENSION : {num_nodes}\n")
            f.write("EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
            for i, (x, y) in enumerate(nodes):
                f.write(f"{i+1} {int(x*scale)} {int(y*scale)}\n")
            f.write("EOF\n")
            
        # 2. Escribir archivo de parámetros .par
        with open(par_file, "w") as f:
            f.write(f"PROBLEM_FILE = {tsp_file}\nOUTPUT_TOUR_FILE = {out_file}\nRUNS = 1\n")

        # 3. Ejecutar LKH
        subprocess.run([executable, par_file], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        # 4. Leer el tour del archivo de salida
        tour = []
        with open(out_file, "r") as f:
            lines = f.readlines()
            start_idx = lines.index("TOUR_SECTION\n") + 1
            for line in lines[start_idx:]:
                node = int(line.strip())
                if node == -1: break
                tour.append(node - 1) # LKH usa 1-indexing
                
        # Calcular coste
        nodes_torch = torch.FloatTensor(nodes).unsqueeze(0)
        tour_torch = torch.LongTensor(tour).unsqueeze(0)
        from problems.tsp.problem_tsp import TSP
        cost, _ = TSP.get_costs(nodes_torch, tour_torch)
        
        return cost.item(), tour

def solve_ortools(nodes, time_limit=1.0):
    """Llama a Google OR-Tools para resolver el TSP usando heurísticas."""
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    # Calcular matriz de distancias escalada a enteros
    scale = 1000000
    dist_matrix = squareform(pdist(nodes, metric='euclidean')) * scale
    dist_matrix = dist_matrix.astype(int).tolist()

    # Configurar OR-Tools
    manager = pywrapcp.RoutingIndexManager(len(nodes), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH) 

    # Asignar tiempo límite usando segundos y nanosegundos para soportar decimales
    search_parameters.time_limit.seconds = int(time_limit)
    search_parameters.time_limit.nanos = int((time_limit % 1) * 1e9)

    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        index = routing.Start(0)
        tour = []
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        
        import torch
        nodes_torch = torch.FloatTensor(nodes).unsqueeze(0)
        tour_torch = torch.LongTensor(tour).unsqueeze(0)
        from problems.tsp.problem_tsp import TSP
        cost, _ = TSP.get_costs(nodes_torch, tour_torch)
        return cost.item(), tour
        
    return float('inf'), []

def solve_pycombinatorial(nodes, algorithm='aco'):
    """Llama a pyCombinatorial para metaheurísticas (ACO, GA)."""
    import numpy as np
    from pyCombinatorial.utils import util
    
    # 1. Convertir coordenadas a array de numpy y construir matriz de distancias
    nodes_array = np.array(nodes)
    distance_matrix = util.build_distance_matrix(nodes_array)
    
    # 2. Seleccionar y ejecutar el algoritmo
    if algorithm == 'aco':
        # Importación dinámica (por si cambia el alias en el futuro)
        try:
            from pyCombinatorial.algorithm import ant_colony_optimization
            tour, _ = ant_colony_optimization(distance_matrix)
        except ImportError:
            from pyCombinatorial.algorithm import aco
            tour, _ = aco(distance_matrix)
            
    elif algorithm == 'ga':
        from pyCombinatorial.algorithm import genetic_algorithm
        # Puedes ajustar (population_size, mutation_rate, generations, etc.) pasándolos como argumentos
        tour, _ = genetic_algorithm(distance_matrix)
    else:
        raise ValueError("Algoritmo pyCombinatorial no soportado")
        
    # El tour devuelto por pyCombinatorial a veces cierra el ciclo repitiendo el nodo inicial.
    # Si el último elemento es igual al primero, lo quitamos para encajar con el formato NCO.
    if tour[0] == tour[-1] and len(tour) > 1:
        tour = tour[:-1]
        
    tour_indices = [int(node) - 1 for node in tour]
    
    # 3. Recalcular el coste exacto usando el formato de tu proyecto (con tensores)
    import torch
    nodes_torch = torch.FloatTensor(nodes).unsqueeze(0)
    tour_torch = torch.LongTensor(tour_indices).unsqueeze(0)
    
    from problems.tsp.problem_tsp import TSP
    cost, _ = TSP.get_costs(nodes_torch, tour_torch)
    
    return cost.item(), tour_indices