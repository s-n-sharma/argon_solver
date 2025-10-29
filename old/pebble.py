import numpy as np
from scipy.optimize import minimize
from enum import Enum
from collections import defaultdict, deque

# --- Section 1: Geometric Primitives and Constraint Classes ---

class Corner(Enum):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3

class Rectangle:
    """A rectangle defined by its top-left corner (x, y), width, and height."""
    def __init__(self, x, y, w, h, name):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.name = name

    def get_params(self):
        return [self.x, self.y, self.w, self.h]

    def set_params(self, params):
        self.x, self.y, self.w, self.h = params

    def get_corner(self, corner_type):
        if corner_type == Corner.TOP_LEFT:
            return np.array([self.x, self.y])
        if corner_type == Corner.TOP_RIGHT:
            return np.array([self.x + self.w, self.y])
        if corner_type == Corner.BOTTOM_LEFT:
            return np.array([self.x, self.y + self.h])
        if corner_type == Corner.BOTTOM_RIGHT:
            return np.array([self.x + self.w, self.y + self.h])
        raise ValueError("Invalid corner type")
    
    def __repr__(self):
        return (f"Rectangle '{self.name}': "
                f"(x={self.x:.2f}, y={self.y:.2f}, w={self.w:.2f}, h={self.h:.2f})")

class CoincidentConstraint:
    def __init__(self, rect1_name, corner1, rect2_name, corner2):
        self.r1, self.c1, self.r2, self.c2 = rect1_name, corner1, rect2_name, corner2
    def get_rect_names(self): return [self.r1, self.r2]
    def calculate_error(self, r): return r[self.r1].get_corner(self.c1) - r[self.r2].get_corner(self.c2)

class FixedDimensionConstraint:
    def __init__(self, rect_name, dimension, value):
        self.r_name, self.dim, self.val = rect_name, dimension, value
    def get_rect_names(self): return [self.r_name]
    def calculate_error(self, r):
        rect = r[self.r_name]
        return np.array([rect.w - self.val] if self.dim == 'width' else [rect.h - self.val])

class FixedPointConstraint:
    def __init__(self, rect_name, corner, point):
        self.r_name, self.c, self.pt = rect_name, corner, np.array(point)
    def get_rect_names(self): return [self.r_name]
    def calculate_error(self, r): return r[self.r_name].get_corner(self.c) - self.pt

# --- Section 2: Pebble Game Helper Class ---
class PebbleGame:
    """A full implementation of the (2,3) pebble game for point-based analysis."""
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.pebbles = {i: 2 for i in range(num_vertices)} # Each point gets 2 pebbles
        self.graph = defaultdict(list)
        self.independent_edges = 0

    def add_edge(self, u, v):
        """Tries to add a constraint edge, returns True if independent."""
        if self.pebbles[u] > 0:
            self._cover_edge(u, v)
            return True
        if self.pebbles[v] > 0:
            self._cover_edge(v, u)
            return True
        
        # No free pebbles; search for one to rearrange
        q = deque([(u, []), (v, [])])
        visited = {u, v}
        
        while q:
            curr, path = q.popleft()
            for neighbor in self.graph[curr]:
                if neighbor not in visited:
                    new_path = path + [(curr, neighbor)]
                    if self.pebbles[neighbor] > 0:
                        # Success: rearrange pebbles along the path
                        self.pebbles[neighbor] -= 1
                        for node_a, node_b in new_path:
                            self.pebbles[node_a] += 1
                            self.pebbles[node_b] -= 1
                        self._cover_edge(u, v)
                        return True
                    visited.add(neighbor)
                    q.append((neighbor, new_path))
        return False # Search failed, edge is dependent

    def _cover_edge(self, pebble_source, other_node):
        """Internal helper to finalize adding an edge."""
        self.pebbles[pebble_source] -= 1
        self.graph[pebble_source].append(other_node)
        self.graph[other_node].append(pebble_source)
        self.independent_edges += 1

# --- Section 3: The Integrated CAD Solver ---

class CADSolver:
    def __init__(self):
        self.rectangles = {}
        self.constraints = []
    
    def add_rectangle(self, rectangle):
        self.rectangles[rectangle.name] = rectangle

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def analyze(self):
        """Analyzes the sketch by modeling it as a system of points."""
        print("--- 🧐 Structural Analysis with Points & Pebble Game ---")
        
        point_map = {}
        point_id_counter = 0
        all_constraint_edges = []
        
        for name in self.rectangles:
            corners = {c: point_id_counter + c.value for c in Corner}
            for corner_enum, point_id in corners.items():
                point_map[(name, corner_enum)] = point_id
            point_id_counter += 4
            
            all_constraint_edges.append((corners[Corner.TOP_LEFT], corners[Corner.BOTTOM_LEFT]))
            all_constraint_edges.append((corners[Corner.TOP_RIGHT], corners[Corner.BOTTOM_RIGHT]))
            all_constraint_edges.append((corners[Corner.TOP_LEFT], corners[Corner.TOP_RIGHT]))
            all_constraint_edges.append((corners[Corner.BOTTOM_LEFT], corners[Corner.BOTTOM_RIGHT]))

        num_fixed_points = 0
        for c in self.constraints:
            if isinstance(c, CoincidentConstraint):
                p1, p2 = point_map.get((c.r1, c.c1)), point_map.get((c.r2, c.c2))
                if p1 is not None and p2 is not None: all_constraint_edges.append((p1, p2))
            if isinstance(c, FixedPointConstraint):
                num_fixed_points += 1
        
        num_points = point_id_counter
        if num_points == 0:
            print("No objects in the sketch.")
            return

        game = PebbleGame(num_points)
        print(f"Analyzing a system of {num_points} points and {len(all_constraint_edges)} constraints...")
        for u, v in all_constraint_edges: game.add_edge(u, v)
        
        independent_constraints = game.independent_edges + (num_fixed_points * 2)
        required_for_rigidity = max(0, 2 * num_points - 3)
        print(f"Found {independent_constraints} independent constraints. Required for rigidity: {required_for_rigidity}.")

        if independent_constraints < required_for_rigidity: print("Status: 🟡 UNDER-CONSTRAINED")
        elif independent_constraints == required_for_rigidity: print("Status: 🟢 WELL-CONSTRAINED")
        else: print("Status: 🔴 OVER-CONSTRAINED")
        print("-" * 52)
        
    def find_rigid_clusters(self):
        """Finds rigid clusters (supernodes) using Tarjan's algorithm for BCCs."""
        print("\n--- 🧩 Decomposing Sketch to Find Rigid Clusters ---")
        _name_to_id = {name: i for i, name in enumerate(self.rectangles.keys())}
        _id_to_name = {i: name for name, i in _name_to_id.items()}
        num_objects = len(self.rectangles)
        if num_objects == 0: return []
        
        adj = defaultdict(list)
        for c in self.constraints:
            names = c.get_rect_names()
            if len(names) == 2 and names[0] in _name_to_id and names[1] in _name_to_id:
                u, v = _name_to_id[names[0]], _name_to_id[names[1]]
                adj[u].append(v); adj[v].append(u)
        
        disc, low, parent, stack, clusters, time = [-1]*num_objects, [-1]*num_objects, [-1]*num_objects, [], [], 0

        def find_bccs(u):
            nonlocal time
            disc[u] = low[u] = time; time += 1
            for v in adj[u]:
                if v == parent[u]: continue
                if disc[v] != -1:
                    low[u] = min(low[u], disc[v])
                    if disc[v] < disc[u]: stack.append((u, v))
                else:
                    parent[v] = u; stack.append((u, v))
                    find_bccs(v)
                    low[u] = min(low[u], low[v])
                    if low[v] >= disc[u]:
                        new_cluster = set()
                        while True:
                            e_u, e_v = stack.pop()
                            new_cluster.add(_id_to_name[e_u]); new_cluster.add(_id_to_name[e_v])
                            if (e_u, e_v) == (u, v): break
                        clusters.append(new_cluster)
        
        for i in range(num_objects):
            if disc[i] == -1: find_bccs(i)

        for i, cluster in enumerate(clusters): print(f"  - Cluster {i+1}: {sorted(list(cluster))}")
        print("-" * 52)
        return clusters
    
    def solve(self):
        """Solves the constraint system numerically."""
        print("\n--- ⚙️  Running Numerical Solver ---")
        initial_guess = []; rect_order = list(self.rectangles.values())
        for r in rect_order: initial_guess.extend(r.get_params())

        def objective_function(params):
            for i, r in enumerate(rect_order): r.set_params(params[i*4 : (i+1)*4])
            return sum(np.sum(c.calculate_error(self.rectangles)**2) for c in self.constraints)

        result = minimize(objective_function, np.array(initial_guess), method='SLSQP')

        if result.success and result.fun < 1e-6:
            print("✅ Solution found successfully!")
            for i, r in enumerate(rect_order): r.set_params(result.x[i*4 : (i+1)*4])
        else:
            print("❌ Solver could not find a valid solution.")
        print("-" * 52)

# --- Section 4: Example Usage ---

# 1. Create the solver and the rectangles
solver = CADSolver()
solver.add_rectangle(Rectangle(10, 10, 50, 50, name="A"))
solver.add_rectangle(Rectangle(100, 20, 50, 50, name="B"))
solver.add_rectangle(Rectangle(200, 30, 50, 50, name="C"))

# 2. Add constraints to form a single, well-defined shape
# Ground the system and fully define rectangle A
solver.add_constraint(FixedPointConstraint("A", Corner.TOP_LEFT, [0, 0]))
solver.add_constraint(FixedDimensionConstraint("A", "width", 50))
solver.add_constraint(FixedDimensionConstraint("A", "height", 80))

# Attach B to A and define its size
solver.add_constraint(CoincidentConstraint("B", Corner.TOP_LEFT, "A", Corner.TOP_RIGHT))
solver.add_constraint(FixedDimensionConstraint("B", "width", 60))
solver.add_constraint(FixedDimensionConstraint("B", "height", 80)) # Same height as A

# Attach C to B and define its size
solver.add_constraint(CoincidentConstraint("C", Corner.TOP_LEFT, "B", Corner.TOP_RIGHT))
solver.add_constraint(FixedDimensionConstraint("C", "width", 40))
solver.add_constraint(FixedDimensionConstraint("C", "height", 80)) # Same height as A

# 3. Run the analysis and solver
solver.analyze()
solver.find_rigid_clusters()

print("\nInitial State:")
for r in solver.rectangles.values(): print(r)
solver.solve()
print("\nFinal State:")
for r in solver.rectangles.values(): print(r)