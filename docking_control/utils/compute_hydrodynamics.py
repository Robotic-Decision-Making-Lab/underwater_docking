import capytaine as cpt
import os
from numpy import inf


cwd = os.path.dirname(__file__)
cpt.set_logging('INFO')
mesh_filepath = cwd + "/../model/bluerov2_custom/bluerov2_custom.stl"
mesh = cpt.load_mesh(mesh_filepath, file_format='stl')

# rov = cpt.FloatingBody(mesh=mesh)
# print(rov.dofs.keys())
# problem = cpt.RadiationProblem(body=rov, omega=inf, free_surface=inf, radiating_dof="Heave")

# solver = cpt.BEMSolver()
# result = solver.solve(problem)
# print(result.added_masses)

mesh.show()
